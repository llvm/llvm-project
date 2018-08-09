// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: 24

package main

import "unsafe"

type a struct {
	a int16
	b int32
	c int8
	d int64
}

func main() {
	println(unsafe.Sizeof(a{}))
}
