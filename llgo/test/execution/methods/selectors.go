// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: F1
// CHECK-NEXT: F2
// CHECK-NEXT: F1
// CHECK-NEXT: F2

package main

type S1 struct{}
type S2 struct {
	S1
}

func (s S1) F1() {
	println("F1")
}

func (s *S2) F2() {
	println("F2")
}

func testUnnamedStructMethods() {
	// Test method lookup on an unnamed struct type.
	var x struct {
		S1
		S2
	}
	x.F1()
	x.F2()
}

func main() {
	var s S2

	// Derive pointer-receiver function.
	f1 := (*S2).F1
	f1(&s)

	f2 := (*S2).F2
	f2(&s)

	testUnnamedStructMethods()
}
