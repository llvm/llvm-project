// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: 10
// CHECK-NEXT: 9

package main

func main() {
	var a [10]int
	b := a[1:]
	println(len(a))
	println(len(b))
}
