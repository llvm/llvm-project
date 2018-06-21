// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: 24
// CHECK-NEXT: 16
// CHECK-NEXT: 0

package main

import "unsafe"

func f1() {
	type T struct {
		a, b, c int
	}
	var t T
	println(unsafe.Sizeof(t))
}

func f2() {
	type T interface{}
	var t T
	t = 1
	println(unsafe.Sizeof(t))
}

func f3() {
	type T struct{}
	var t T
	println(unsafe.Sizeof(t))
}

func main() {
	f1()
	f2()
	f3()
}
