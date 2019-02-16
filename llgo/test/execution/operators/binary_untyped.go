// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: true

package main

func f1(b bool) bool {
	return b
}

func main() {
	x := false
	y := x
	x = !y
	println(x || y)
}
