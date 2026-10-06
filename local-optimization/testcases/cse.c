// Test case - Common subexpression elimination

int compute(int b, int c){
    int a = b + c;
    int d = b + c; // same expression -> reuse a

    return a * d;
}

// Expected (checked by run_tests.sh): b + c is computed once.
// CHECK:     [[A:%.*]] = add nsw i32 %b, %c
// CHECK-NOT: = add
// CHECK:     mul nsw i32 [[A]], [[A]]
