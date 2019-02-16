// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: 5
// CHECK-NEXT: 0

package main

func main() {
	a := make([]int, 10)
	b := make([]int, 10)
	for i, _ := range b {
		b[i] = 1
	}
	println(copy(a[:5], b)) // expect 5
	println(a[5])           // expect 0
}
