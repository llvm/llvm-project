// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: 0

package main

func blah() []int {
	return make([]int, 1)
}

func main() {
	println(blah()[0])
}
