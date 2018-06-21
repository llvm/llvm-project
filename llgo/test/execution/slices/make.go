// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: 0 0
// CHECK-NEXT: 1 0
// CHECK-NEXT: 2 0
// CHECK-NEXT: 3 0
// CHECK-NEXT: 4 0
// CHECK-NEXT: 5 666
// CHECK-NEXT: 6 0
// CHECK-NEXT: 7 0
// CHECK-NEXT: 8 0
// CHECK-NEXT: 9 0

package main

func main() {
	x := make([]int, 10)
	x[5] = 666
	for i, val := range x {
		println(i, val)
	}
}
