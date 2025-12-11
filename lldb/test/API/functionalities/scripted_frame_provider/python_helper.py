"""
Sample Python module to demonstrate Python source display in scripted frames.
"""


def compute_fibonacci(n):
    """Compute the nth Fibonacci number."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b


def process_data(data):
    """Process some data and return result."""
    result = []
    for item in data:
        if isinstance(item, int):
            result.append(item * 2)
        elif isinstance(item, str):
            result.append(item.upper())
    return result


def main():
    """Main entry point for testing."""
    fib_10 = compute_fibonacci(10)
    data = [1, 2, "hello", 3, "world"]
    processed = process_data(data)
    return fib_10, processed


if __name__ == "__main__":
    main()
