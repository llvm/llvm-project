// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: 0
// CHECK-NEXT: 2
// CHECK-NEXT: 4

package main

func main() {
	x := new(int)
	println(*x)
	*x = 2
	println(*x)
	*x = *x * *x
	println(*x)
}
