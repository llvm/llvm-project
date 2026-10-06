#include<stdio.h>

// Test case - Strength reduction by replacing
// multiplication by powers of 2 with left shift

int main(){
    int a=3,b=4,c,d,e,f;

    d = 2*c; // strength reduction c << 1
    e = f*8; // strength reduction f << 3

    return a;
}

// ---------------------------------------------------------------------------
// Expected results, checked by run_tests.sh with FileCheck.
//
// TEST-OPTS: strength-reduction
// ONLY-LABEL: define {{.*}} @main(
// ONLY:       [[C:%.*]] = load i32, ptr %c,
// ONLY-NEXT:  %mul = shl nsw i32 [[C]], 1
// ONLY:       [[F:%.*]] = load i32, ptr %f,
// ONLY-NEXT:  %mul1 = shl nsw i32 [[F]], 3
//
// All optimizations: d and e are never read, so the shifts are dead code.
// ALL-LABEL:  define {{.*}} @main(
// ALL-NEXT:   entry:
// ALL-NEXT:     ret i32 3
