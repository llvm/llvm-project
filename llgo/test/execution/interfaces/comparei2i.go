// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: true
// CHECK-NEXT: true

package main

import "unsafe"

type I interface {
	X()
}

type T int

func (t T) X() {
}

func main() {
	var highbit uint32 = 1 << 31
	var pos0 float32 = 0
	var neg0 float32 = *(*float32)(unsafe.Pointer(&highbit))
	var i1 interface{} = pos0
	var i2 interface{} = neg0
	println(i1 == i2)
	var i3 interface{} = T(123)
	var i4 I = T(123)
	println(i3 == i4)
}
