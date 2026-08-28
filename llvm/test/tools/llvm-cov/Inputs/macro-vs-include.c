// Test that --ignore-branch-in-macro distinguishes macros from #include

// Header with branch
static inline int header_func(int x) {
    if (x > 10) {
        return x * 2;
    }
    return x;
}

// Macro with branch
#define CLAMP(x, lo, hi) ((x) < (lo) ? (lo) : (x) > (hi) ? (hi) : (x))

int test_func(int val) {
    // Macro branches - should be filtered
    int clamped = CLAMP(val, 1, 100);

    // #include branch - should be preserved
    int h = header_func(val);

    // Source branch - always counted
    if (val > 0) {
        return clamped + h;
    }
    return 0;
}

int main() {
    test_func(1);
    test_func(-1);
    test_func(50);
    test_func(150);
    return 0;
}
