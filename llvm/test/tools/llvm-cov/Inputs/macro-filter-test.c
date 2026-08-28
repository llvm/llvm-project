// Simple test for --ignore-branch-in-macro
// This file is compiled to generate test input data

#define MY_ABS(x) ((x) >= 0 ? (x) : -(x))
#define MY_MAX(a, b) ((a) > (b) ? (a) : (b))

int test_function(int val) {
    // Macro branches - should be filtered with --ignore-branch-in-macro
    int abs_val = MY_ABS(val);
    int max_val = MY_MAX(val, 5);

    // Source-level branches - always counted
    if (val > 0) {
        return abs_val + max_val;
    }
    return 0;
}

int main() {
    test_function(1);
    test_function(-1);
    test_function(10);
    return 0;
}
