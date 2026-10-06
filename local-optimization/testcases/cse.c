// Test case - Common subexpression elimination

int compute(int b, int c){
    int a = b + c;
    int d = b + c;   // same expression as a -> reuse a
    int e = c + b;   // same again (addition is commutative)

    return a * d * e;
}

// ---------------------------------------------------------------------------
// Expected results, checked by run_tests.sh with FileCheck.
//
// TEST-OPTS: constprop;cse
// Constant propagation first turns the loads of b and c into the arguments
// %b and %c, so the three additions become identical.
// ONLY-LABEL: define {{.*}} @compute(
// ONLY:       [[A:%.*]] = add nsw i32 %b, %c
// ONLY-NOT:   = add
// ONLY:       [[M:%.*]] = mul nsw i32 [[A]], [[A]]
// ONLY:       mul nsw i32 [[M]], [[A]]
//
// ALL-LABEL:  define {{.*}} @compute(
// ALL-NEXT:   entry:
// ALL-NEXT:     [[A:%.*]] = add nsw i32 %b, %c
// ALL-NEXT:     [[M:%.*]] = mul nsw i32 [[A]], [[A]]
// ALL-NEXT:     [[R:%.*]] = mul nsw i32 [[M]], [[A]]
// ALL-NEXT:     ret i32 [[R]]
