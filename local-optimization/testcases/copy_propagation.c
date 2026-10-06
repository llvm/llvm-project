#include<stdio.h>
//Test case - Copy propagation

int main(){
	int b=4, c,d,e;	
	c=d;//copy propagation	
	e = c+b; 
	return e;
}

// ---------------------------------------------------------------------------
// Expected results, checked by run_tests.sh with FileCheck.
// (d is uninitialized, so its value stays an unknown load from %d.)
//
// TEST-OPTS: constprop
// "e = c + b" becomes "e = d + 4": c is replaced by d's value, b by 4.
// ONLY-LABEL: define {{.*}} @main(
// ONLY:       [[D:%.*]] = load i32, ptr %d,
// ONLY:       [[E:%.*]] = add nsw i32 [[D]], 4
// ONLY:       ret i32 [[E]]
//
// ALL-LABEL:  define {{.*}} @main(
// ALL-NEXT:   entry:
// ALL-NEXT:     %d = alloca i32
// ALL-NEXT:     [[D:%.*]] = load i32, ptr %d,
// ALL-NEXT:     [[E:%.*]] = add nsw i32 [[D]], 4
// ALL-NEXT:     ret i32 [[E]]
