import sys

def review_openmp_code(code):
    # Simulated review (replace with real model call if needed)
    mocked_response = """
🤖 Mock Review:
- No obvious race conditions detected.
- Verify the use of #pragma omp parallel for directives.
- Consider using reduction clauses for shared variables involved in summations.
- Use critical sections or atomic operations where appropriate to prevent data races.
- Overall, your OpenMP usage looks good with these suggestions.
"""
    return mocked_response

def main():
    if len(sys.argv) != 2:
        print("Usage: python cli.py <code_file.c>")
        return

    filepath = sys.argv[1]

    try:
        with open(filepath, 'r') as f:
            code = f.read()
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return

    print("\n📝 OpenMP Review Bot Running (Mock Mode)...\n")
    result = review_openmp_code(code)
    print("🔍 AI Review Result:\n")
    print(result)

if __name__ == "__main__":
    main()
