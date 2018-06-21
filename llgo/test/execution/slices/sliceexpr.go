// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: a
// CHECK-NEXT: 0 2
// CHECK-NEXT: 1 3
// CHECK-NEXT: b
// CHECK-NEXT: 0 3
// CHECK-NEXT: 1 4
// CHECK-NEXT: c
// CHECK-NEXT: 0 1
// CHECK-NEXT: 1 2
// CHECK-NEXT: d
// CHECK-NEXT: 0 1
// CHECK-NEXT: 1 2
// CHECK-NEXT: 2 3
// CHECK-NEXT: 3 4

package main

func main() {
	x := []int{1, 2, 3, 4}
	println("a")
	for i, val := range x[1:3] {
		println(i, val)
	}
	println("b")
	for i, val := range x[2:] {
		println(i, val)
	}
	println("c")
	for i, val := range x[:2] {
		println(i, val)
	}
	println("d")
	for i, val := range x[:] {
		println(i, val)
	}
}
