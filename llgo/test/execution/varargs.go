// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: 3
// CHECK-NEXT: 123
// CHECK-NEXT: 456
// CHECK-NEXT: 789
// CHECK-NEXT: 4
// CHECK-NEXT: 123
// CHECK-NEXT: 456
// CHECK-NEXT: 789
// CHECK-NEXT: 101112
// CHECK-NEXT: 3
// CHECK-NEXT: 1
// CHECK-NEXT: 2
// CHECK-NEXT: 3

package main

func p(i ...int) {
	println(len(i))
	for j := 0; j < len(i); j++ {
		println(i[j])
	}
}

func main() {
	p(123, 456, 789)
	p(123, 456, 789, 101112)
	p([]int{1, 2, 3}...)
}
