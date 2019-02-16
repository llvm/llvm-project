// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: true
// CHECK-NEXT: false

package main

func main() {
	var x interface{} = 123
	println(x == 123)
	println(x != 123)
}
