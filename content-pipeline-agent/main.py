# -*- coding: utf-8 -*-
from typing import List, Optional

from crewai.flow.flow import Flow, listen, start, router, and_, or_
from crewai import Agent
from crewai import LLM
from pydantic import BaseModel

from tools import web_search_tool
from seo_crew import SeoCrew
from virality_crew import ViralityCrew


class BlogPost(BaseModel):
    title: str
    subtitle: str
    sections: List[str]


class Tweet(BaseModel):
    content: str
    hashtags: str


class LinkedInPost(BaseModel):
    hook: str
    content: str
    call_to_action: str


class Score(BaseModel):
    score: int = 0
    reason: str = ""


class ContentPipelineState(BaseModel):
    # Inputs
    content_type: str = ""
    topic: str = ""

    # Internal
    max_length: int = 0
    score: Optional[Score] = None
    research: str = ""

    # Content
    blog_post: Optional[BlogPost] = None
    tweet: Optional[Tweet] = None
    linkedin_post: Optional[LinkedInPost] = None


class ContentPipelineFlow(Flow[ContentPipelineState]):

    @start()
    def init_content_pipeline(self):
        if self.state.content_type not in ["tweet", "blog", "linkedin"]:
            raise ValueError("The Content type is wrong!")

        if self.state.topic == "":
            raise ValueError("The topic can't be blank.")

        if self.state.content_type == "tweet":
            self.state.max_length = 150
        elif self.state.content_type == "blog":
            self.state.max_length = 800
        elif self.state.content_type == "linkedin":
            self.state.max_length = 500

    @listen(init_content_pipeline)
    def conduct_research(self):
        researcher = Agent(
            role="Head Researcher",
            backstory="You're like a digital detective who loves digging up fascinating facts and insights. You have a knack for finding the good stuff that others miss.",
            goal=f"Find the most interesting and useful info about {self.state.topic}",
            tools=[web_search_tool],
        )
        # NOTE: crewai.Agent.kickoff 반환형에 따라 필요시 .output / .raw 등으로 조정
        self.state.research = researcher.kickoff(
            messages=f"Find the most interesting and useful info about {self.state.topic}"
        )
        return True

    @router(conduct_research)
    def conduct_research_router(self):
        content_type = self.state.content_type
        if content_type == "blog":
            return "make_blog"
        elif content_type == "tweet":
            return "make_tweet"
        else:
            return "make_linkedin_post"

    @listen(or_("make_blog", "remake_blog"))
    def handle_make_blog(self):
        blog_post = self.state.blog_post
        llm = LLM(model="openai/gpt-4o-mini", response_format=BlogPost)

        if blog_post is None:
            result = llm.call(
                f"""
Make a blog post with SEO practices on the topic {self.state.topic} using the following research:

<research>
=======================
{self.state.research}
=======================
</research>
"""
            )
        else:
            result = llm.call(
                f"""
You wrote this blog post on {self.state.topic}, but it does not have a good SEO score because of {self.state.score.reason}. Improve it.

<blog post>
{self.state.blog_post.model_dump_json()}
</blog post>

Use the following research.

<research>
=======================
{self.state.research}
=======================
</research>
"""
            )

        self.state.blog_post = BlogPost.model_validate_json(result)

    @listen(or_("make_tweet", "remake_tweet"))
    def handle_make_tweet(self):
        tweet = self.state.tweet
        llm = LLM(model="openai/gpt-4o-mini", response_format=Tweet)

        if tweet is None:
            result = llm.call(
                f"""
Make a tweet that can go viral on the topic {self.state.topic} using the following research:

<research>
=======================
{self.state.research}
=======================
</research>
"""
            )
        else:
            result = llm.call(
                f"""
You wrote this tweet on {self.state.topic}, but it does not have a good virality score because of {self.state.score.reason}. Improve it.

<tweet>
{self.state.tweet.model_dump_json()}
</tweet>

Use the following research.

<research>
=======================
{self.state.research}
=======================
</research>
"""
            )

        self.state.tweet = Tweet.model_validate_json(result)

    @listen(or_("make_linkedin_post", "remake_linkedin_post"))
    def handle_make_linkedin_post(self):
        linkedin_post = self.state.linkedin_post
        # FIX: response_format를 LinkedInPost로 교정
        llm = LLM(model="openai/gpt-4o-mini", response_format=LinkedInPost)

        if linkedin_post is None:
            result = llm.call(
                f"""
Make a linkedin post that can go viral on the topic {self.state.topic} using the following research:

<research>
=======================
{self.state.research}
=======================
</research>
"""
            )
        else:
            result = llm.call(
                f"""
You wrote this linkedin post on {self.state.topic}, but it does not have a good virality score because of {self.state.score.reason}. Improve it.

<linkedin_post>
{self.state.linkedin_post.model_dump_json()}
</linkedin_post>

Use the following research.

<research>
=======================
{self.state.research}
=======================
</research>
"""
            )

        self.state.linkedin_post = LinkedInPost.model_validate_json(result)

    @listen(handle_make_blog)
    def check_seo(self):
        result = SeoCrew().crew().kickoff(
            inputs={
                "topic": self.state.topic,
                "blog_post": self.state.blog_post.model_dump_json(),
            }
        )
        # result.pydantic의 타입이 dict라면 Score(**result.pydantic)로 감싸도 됨
        self.state.score = result.pydantic  # 또는: Score(**result.pydantic)

    @listen(or_(handle_make_tweet, handle_make_linkedin_post))
    def check_virality(self):
        content_json = (
            self.state.tweet.model_dump_json()
            if self.state.content_type == "tweet"
            else self.state.linkedin_post.model_dump_json()
        )
        result = ViralityCrew().crew().kickoff(
            inputs={
                "topic": self.state.topic,
                "content_type": self.state.content_type,
                "content": content_json,
            }
        )
        self.state.score = result.pydantic  # 또는: Score(**result.pydantic)

    @router(or_(check_seo, check_virality))
    def score_router(self):
        content_type = self.state.content_type
        score = self.state.score

        print(score)

        if score and score.score >= 7:
            return "checks_passed"
        else:
            if content_type == "blog":
                return "remake_blog"
            elif content_type == "linkedin":
                return "remake_linkedin_post"
            else:
                return "remake_tweet"

    @listen("checks_passed")
    def finalize_content(self):
        print("Finalizing content")


if __name__ == "__main__":
    flow = ContentPipelineFlow()

    flow.kickoff(
        inputs={
            "content_type": "tweet",
            "topic": "AI 깐부 세트",
        },
    )

    flow.plot()
