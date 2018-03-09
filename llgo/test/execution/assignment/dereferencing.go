// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: 123

package main

func main() {
	var x int
	px := &x
	*px = 123
	println(x)
}
