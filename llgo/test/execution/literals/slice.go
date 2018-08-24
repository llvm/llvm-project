// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: abc
// CHECK-NEXT: 123
// CHECK-NEXT: abc
// CHECK-NEXT: 123

package main

func main() {
	x := []string{"abc", "123"}
	println(x[0])
	println(x[1])

	// Elements are composite literals, so the '&' can be elided.
	type S struct{ string }
	y := []*S{{"abc"}, {"123"}}
	println(y[0].string)
	println(y[1].string)
}
