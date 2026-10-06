#include<stdio.h>
//Test case - redundant assignment elimination
int main(){
	int a=3,b=4,c;
	a=3; //redundant assignment elimination
	b=4; // 
	c = a+b;
	return a;
}

// ---------------------------------------------------------------------------
// Expected results, checked by run_tests.sh with FileCheck.
//
// TEST-OPTS: dce
// DCE alone: "a=3; b=4;" are redundant assignments, and c is never read, so
// "c = a+b" goes too. Then b is never read either.
// ONLY-LABEL: define {{.*}} @main(
// ONLY-NOT:   %b =
// ONLY-NOT:   %c =
// ONLY:       store i32 3, ptr %a,
// ONLY-NOT:   store
// ONLY:       [[A:%.*]] = load i32, ptr %a,
// ONLY-NEXT:  ret i32 [[A]]
//
// All optimizations: constant propagation turns "return a" into "return 3".
// ALL-LABEL:  define {{.*}} @main(
// ALL-NEXT:   entry:
// ALL-NEXT:     ret i32 3
