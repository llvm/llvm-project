#include<stdio.h>
//Test case - redundant assignment elimination
int main(){
	int a=3,b=4,c;
	a=3; //redundant assignment elimination
	b=4; // 
	c = a+b;
	return a;
}

// Expected (checked by run_tests.sh): the second a=3 and b=4 are gone,
// a+b is folded to 7.
// CHECK:      store i32 3, ptr %a
// CHECK-NEXT: store i32 4, ptr %b
// CHECK-NEXT: store i32 7, ptr %c
// CHECK-NEXT: ret i32 3
