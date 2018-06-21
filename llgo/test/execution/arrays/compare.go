// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: false
// CHECK-NEXT: true
// CHECK-NEXT: false

package main

func main() {
	a := [...]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	b := [...]int{10, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	c := [...]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

	println(a == b)
	println(a == c)
	println(b == c)
}
