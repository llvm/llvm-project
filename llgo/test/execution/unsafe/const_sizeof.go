// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: 8
// CHECK-NEXT: 8

package main

import "unsafe"

const ptrSize = unsafe.Sizeof((*byte)(nil))

var x [ptrSize]int

func main() {
	println(ptrSize)
	println(len(x))
}
