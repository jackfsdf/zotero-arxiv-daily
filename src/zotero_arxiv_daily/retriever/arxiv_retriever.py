from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from tempfile import TemporaryDirectory
import feedparser
from urllib.request import urlretrieve
from tqdm import tqdm
import os
import time
from loguru import logger
from datetime import datetime, timedelta

@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")
    
    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        client = arxiv.Client(num_retries=10, delay_seconds=10)
        
        # 第一部分：原有的 RSS 分类检索
        category_papers = self._retrieve_by_category(client)
        
        # 第二部分：新增的关键词检索（如果配置了关键词）
        keyword_papers = self._retrieve_by_keywords(client)
        
        # 合并两部分结果，去重
        all_papers = self._merge_papers(category_papers + keyword_papers)
        
        if self.config.executor.debug:
            all_papers = all_papers[:10]
        
        return all_papers
    
    def _retrieve_by_category(self, client: arxiv.Client) -> list[ArxivResult]:
        """原有的 RSS 分类检索逻辑"""
        query = '+'.join(self.config.source.arxiv.category)
        feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
        
        if 'Feed error for query' in feed.feed.title:
            logger.warning(f"Invalid ARXIV_QUERY: {query}. Skipping category retrieval.")
            return []
        
        all_paper_ids = [
            i.id.removeprefix("oai:arXiv.org:") for i in feed.entries 
            if i.get("arxiv_announce_type", "new") == 'new'
        ]
        
        if not all_paper_ids:
            return []
        
        raw_papers = []
        bar = tqdm(total=len(all_paper_ids), desc="Fetching category papers")
        
        for idx, i in enumerate(range(0, len(all_paper_ids), 20)):
            if idx > 0:
                time.sleep(5)
            try:
                search = arxiv.Search(id_list=all_paper_ids[i:i+20])
                batch = list(client.results(search))
                bar.update(len(batch))
                raw_papers.extend(batch)
            except Exception as e:
                logger.warning(f"Batch {idx} failed: {e}, retrying after 30s...")
                time.sleep(30)
                try:
                    search = arxiv.Search(id_list=all_paper_ids[i:i+20])
                    batch = list(client.results(search))
                    bar.update(len(batch))
                    raw_papers.extend(batch)
                except Exception as e2:
                    logger.error(f"Batch {idx} failed again: {e2}, skipping.")
                    bar.update(len(all_paper_ids[i:i+20]))
        
        bar.close()
        logger.info(f"Retrieved {len(raw_papers)} papers from category search")
        return raw_papers
    
    def _retrieve_by_keywords(self, client: arxiv.Client) -> list[ArxivResult]:
        """改进的关键词检索逻辑"""
        # 检查是否配置了关键词
        if not hasattr(self.config.source.arxiv, 'keywords') or not self.config.source.arxiv.keywords:
            return []
        
        # 构建关键词查询（OR关系）
        keywords = self.config.source.arxiv.keywords
        keyword_conditions = []
        
        for keyword in keywords:
            # 处理带空格的关键词（加引号）
            if ' ' in keyword:
                keyword_conditions.append(f'all:"{keyword}"')  # 使用all字段同时搜索标题和摘要
            else:
                keyword_conditions.append(f'all:{keyword}')
        
        # 关键词部分：所有字段中包含任意关键词
        keyword_query = ' OR '.join(keyword_conditions)
        
        # 时间限制：最近2天，使用明确的时间范围（避免通配符*）
        from datetime import datetime, timedelta
        
        # 起始时间：2天前的00:00
        start_date = (datetime.now() - timedelta(days=2)).strftime("%Y%m%d")
        start_datetime = f"{start_date}0000"
        
        # 结束时间：当前时间的23:59（或使用当前日期）
        end_date = datetime.now().strftime("%Y%m%d")
        end_datetime = f"{end_date}2359"  # 明确的结束时间
        
        date_query = f"submittedDate:[{start_datetime} TO {end_datetime}]"
        
        # 【关键修改】不使用cat:cs.*通配符，改为显式列出常见CS子领域
        # 可以根据你的需求调整这个列表
        cs_categories = [
            "cs.AI", "cs.LG", "cs.CV", "cs.CL", "cs.RO", 
            "cs.IR", "cs.MM",   
            "cs.MA"  
        ]
        
        # 构建分类查询（OR关系）
        category_conditions = [f"cat:{cat}" for cat in cs_categories]
        category_query = '(' + ' OR '.join(category_conditions) + ')'
        
        # 组合完整查询
        full_query = f"({keyword_query}) AND {category_query} AND {date_query}"
        
        logger.info(f"Keyword search query: {full_query}")
        
        try:
            search = arxiv.Search(
                query=full_query,
                max_results=100,  # 直接设置总结果数
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            # 获取所有结果
            results = list(client.results(search))
            logger.info(f"Retrieved {len(results)} papers from keyword search")
            
            return results
                
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    def _merge_papers(self, papers: list[ArxivResult]) -> list[ArxivResult]:
        """按论文ID去重合并"""
        seen = set()
        unique_papers = []
        
        for paper in papers:
            if paper.entry_id not in seen:
                seen.add(paper.entry_id)
                unique_papers.append(paper)
        
        logger.info(f"Total unique papers after merging: {len(unique_papers)}")
        return unique_papers

    def convert_to_paper(self, raw_paper: ArxivResult) -> Paper | None:
        try:
            title = raw_paper.title
            authors = [a.name for a in raw_paper.authors]
            abstract = raw_paper.summary
            pdf_url = raw_paper.pdf_url
            full_text = extract_text_from_pdf(raw_paper)
            if full_text is None:
                full_text = extract_text_from_tar(raw_paper)
            return Paper(
                source=self.name,
                title=title,
                authors=authors,
                abstract=abstract,
                url=raw_paper.entry_id,
                pdf_url=pdf_url,
                full_text=full_text
            )
        except Exception as e:
            logger.warning(f"Failed to convert paper '{raw_paper.title}': {e}")
            return None

def extract_text_from_pdf(paper: ArxivResult) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        if paper.pdf_url is None:
            logger.warning(f"No PDF URL available for {paper.title}")
            return None
        urlretrieve(paper.pdf_url, path)
        try:
            full_text = extract_markdown_from_pdf(path)
        except Exception as e:
            logger.warning(f"Failed to extract full text of {paper.title} from pdf: {e}")
            full_text = None
        return full_text

def extract_text_from_tar(paper: ArxivResult) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        source_url = paper.source_url()
        if source_url is None:
            logger.warning(f"No source URL available for {paper.title}")
            return None
        urlretrieve(source_url, path)
        try:
            file_contents = extract_tex_code_from_tar(path, paper.entry_id)
            if "all" not in file_contents:
                logger.warning(f"Failed to extract full text of {paper.title} from tar: Main tex file not found.")
                return None
            full_text = file_contents["all"]
        except Exception as e:
            logger.warning(f"Failed to extract full text of {paper.title} from tar: {e}")
            full_text = None
        return full_text
