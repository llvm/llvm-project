// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: 3 3 6
// CHECK-NEXT: abc123

package main

func main() {
	a := "abc"
	b := "123"
	c := a + b
	println(len(a), len(b), len(c))
	println(c)
}
