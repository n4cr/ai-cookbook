import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Tuple
import json

# Define Pydantic models for the analysis of customer reviews
class ReviewInsight(BaseModel):
    phrase: str = Field(description="A key phrase extracted from the review that holds significant insight.")
    sentiment: str = Field(description="The sentiment expressed regarding the key phrase, e.g., positive, negative, neutral.")

class ActionableItem(BaseModel):
    action: str = Field(description="A suggested action or area for improvement identified from the review.")
    importance: str = Field(description="The level of importance or urgency of the action, e.g., high, medium, low.")

class ReviewAnalysis(BaseModel):
    overall_sentiment: str = Field(description="The overall sentiment of the review, e.g., positive, negative, neutral.")
    insights: List[ReviewInsight] = Field(default_factory=list, description="List of key insights extracted from the review.")
    actionable_items: List[ActionableItem] = Field(default_factory=list, description="List of actionable items or suggestions for improvement.")

# Patch the OpenAI client to add response_model support
client = instructor.patch(OpenAI())

def analyze_customer_review(review_text: str) -> ReviewAnalysis:
    """Analyzes a customer review to extract sentiment, key insights, and actionable items."""
    return client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {
                "role": "user",
                "content": f"Analyze the following customer review for sentiment, key insights, and actionable items: {review_text}"
            }
        ],
        response_model=ReviewAnalysis
    )

if __name__ == "__main__":
    # Example customer review
    customer_review = """
    I recently purchased the SmartWidget 3000 and had high expectations based on the ads. However, I encountered several issues.
    The setup process was far more complicated than anticipated, and the user manual wasn't helpful. Once I got it running,
    the performance was decent, but it occasionally overheats with extended use. On the plus side, customer service was responsive
    and helped me troubleshoot some problems. I hope they can improve the manual and address the overheating issue in future models.
    """
    with open("customer_review/review.txt", "r") as file:
        customer_review = file.read()

    # Analyze the customer review
    analysis = analyze_customer_review(customer_review)
    
    # Print the analysis in JSON format for readability
    print(json.dumps(analysis.model_dump(), indent=2))

