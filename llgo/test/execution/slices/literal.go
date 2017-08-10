// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: 1
// CHECK-NEXT: 2
// CHECK-NEXT: 3

package main

func main() {
	x := []int{1, 2, 3}
	for i := 0; i < len(x); i++ {
		println(x[i])
	}
}
