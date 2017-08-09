// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: 0 0 0
// CHECK-NEXT: 1 1 1
// CHECK-NEXT: 2 2 2
// CHECK-NEXT: 3 0 0
// CHECK-NEXT: 4 4 4
// CHECK-NEXT: 0 10
// CHECK-NEXT: 1 20
// CHECK-NEXT: 2 30

package main

func main() {
	a := [...]int{1: 1, 2: 2, 4: 4}
	for i, val := range a {
		println(i, val, a[i])
	}
	for i, val := range [...]int{10, 20, 30} {
		println(i, val)
	}
}
