#include<stdio.h>
// Test case - Dead code elimination

int main(){
    int x = 10, y = 20, unused;

    unused = x * y + 5;   // never used, no side effects -> removed
    printf("Hello\n");    // result unused, but it prints -> kept

    return x;
}

// ---------------------------------------------------------------------------
// Expected results, checked by run_tests.sh with FileCheck.
//
// TEST-OPTS: dce
// ONLY-LABEL: define {{.*}} @main(
// ONLY-NOT:   %unused
// ONLY-NOT:   = mul
// ONLY:       call {{.*}} @printf(
// ONLY:       ret i32
//
// ALL-LABEL:  define {{.*}} @main(
// ALL-NEXT:   entry:
// ALL-NEXT:     call {{.*}} @printf(
// ALL-NEXT:     ret i32 10
