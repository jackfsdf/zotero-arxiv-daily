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
from loguru import logger

@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")
    
    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        client = arxiv.Client(num_retries=10, delay_seconds=10)
        query = '+'.join(self.config.source.arxiv.category)
        
        # Get the latest paper from arxiv rss feed
        feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
        if 'Feed error for query' in feed.feed.title:
            raise Exception(f"Invalid ARXIV_QUERY: {query}.")
        
        # 获取所有新论文的ID
        all_paper_ids = [i.id.removeprefix("oai:arXiv.org:") for i in feed.entries 
                         if i.get("arxiv_announce_type", "new") == 'new']
        
        if self.config.executor.debug:
            all_paper_ids = all_paper_ids[:10]
        
        # Get full information of each paper from arxiv api
        raw_papers = []
        bar = tqdm(total=len(all_paper_ids))
        
        for i in range(0, len(all_paper_ids), 20):
            search = arxiv.Search(id_list=all_paper_ids[i:i+20])
            batch = list(client.results(search))
            
            # 如果有关键词配置，进行过滤
            if hasattr(self.config.source.arxiv, 'keywords') and self.config.source.arxiv.keywords:
                filtered_batch = []
                for paper in batch:
                    if self._contains_keywords(paper.title, paper.summary):
                        filtered_batch.append(paper)
                batch = filtered_batch
            
            bar.update(len(batch))
            raw_papers.extend(batch)
        
        bar.close()
        return raw_papers
    
    def _contains_keywords(self, title: str, abstract: str) -> bool:
        """
        检查标题或摘要中是否包含配置的关键词（OR关系）
        """
        if not hasattr(self.config.source.arxiv, 'keywords') or not self.config.source.arxiv.keywords:
            return True
        
        # 将标题和摘要合并，转为小写以便不区分大小写比较
        text = (title + " " + abstract).lower()
        
        # 检查是否包含任意一个关键词（OR关系）
        for keyword in self.config.source.arxiv.keywords:
            if keyword.lower() in text:
                logger.debug(f"Paper matched keyword '{keyword}': {title[:50]}...")
                return True
        
        return False

    def convert_to_paper(self, raw_paper: ArxivResult) -> Paper:
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
