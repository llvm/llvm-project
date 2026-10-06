// Test case - Constant propagation + constant folding

int compute ()
{
    int result = 0;
    int a = 2;
    int b = 3;
    int c = 4 + a + b; // c=9

    result += a; // res = 0 + 2
    result += b; // 2 + 3
    result *= c; // 5 * 9
    result /= 2; // 45/2=22

    a = result;

    return result;
}

// ---------------------------------------------------------------------------
// Expected results, checked by run_tests.sh with FileCheck.
//
// TEST-OPTS: constprop;instcombine
// Every intermediate value becomes a constant.
// ONLY-LABEL: define {{.*}} @compute(
// ONLY:       store i32 9, ptr %c,
// ONLY-NEXT:  store i32 2, ptr %result,
// ONLY-NEXT:  store i32 5, ptr %result,
// ONLY-NEXT:  store i32 45, ptr %result,
// ONLY-NEXT:  store i32 22, ptr %result,
// ONLY-NEXT:  store i32 22, ptr %a,
// ONLY-NEXT:  ret i32 22
//
// ALL-LABEL:  define {{.*}} @compute(
// ALL-NEXT:   entry:
// ALL-NEXT:     ret i32 22
