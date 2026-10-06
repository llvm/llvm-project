#include<stdio.h>
//Test case - Copy propagation

int main(){
	int b=4, c,d,e;	
	c=d;//copy propagation	
	e = c+b; 
	return e;
}

// Expected (checked by run_tests.sh): e = c + b  became  e = d + 4.
// CHECK:      [[D:%.*]] = load i32, ptr %d
// CHECK:      [[E:%.*]] = add nsw i32 [[D]], 4
// CHECK:      ret i32 [[E]]
