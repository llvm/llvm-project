// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: 0
// CHECK-NEXT: 4
// CHECK-NEXT: 8
// CHECK-NEXT: 16

package main

import "unsafe"

type S struct {
	a int16
	b int32
	c int8
	d int64
}

func main() {
	var s S
	println(unsafe.Offsetof(s.a))
	println(unsafe.Offsetof(s.b))
	println(unsafe.Offsetof(s.c))
	println(unsafe.Offsetof(s.d))
}
