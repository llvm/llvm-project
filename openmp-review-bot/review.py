import openai

# ✅ Use your actual API key securely (prefer env vars in production)
client = openai.OpenAI(api_key="YOUR_API_KEY_HERE")
code_to_review = """
#include <omp.h>
#include <stdio.h>

int main() {
    int sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < 100; i++) {
        sum += i;
    }
    printf("Sum = %d\\n", sum);
    return 0;
}
"""

def request_review(model_name):
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": f"Please review the following OpenMP C code for correctness, performance, and race conditions:\n\n{code_to_review}"
                }
            ]
        )
        return response.choices[0].message.content
    except openai.RateLimitError as e:
        print(f"⚠️ RateLimitError: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

# ✅ Try GPT-4o first, fallback to GPT-3.5 if quota exceeded
review = request_review("gpt-4o")
if not review:
    print("⚠️ Falling back to GPT-3.5-turbo...")
    review = request_review("gpt-3.5-turbo")

# ✅ Display the result
if review:
    print("📋 AI Review:")
    print(review)
else:
    print("❌ Could not get a review from OpenAI API.")
