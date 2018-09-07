// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: 246
// CHECK-NEXT: 123 true false

// vim: set ft=go :

package main

func test() func() int {
	return blah
}

func blah() int {
	return 123
}

func sret() (int, bool, bool) {
	return 123, true, false
}

func main() {
	f := test()
	println(2 * f())
	a, b, c := sret()
	println(a, b, c)
}
