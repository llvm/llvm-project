// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: A
// CHECK-NEXT: B

package main

type BI interface {
	B()
}

type AI interface {
	A()
	BI
}

type S struct{}

func (s S) A() {
	println("A")
}

func (s S) B() {
	println("B")
}

func main() {
	var ai AI = S{}
	ai.A()
	ai.B()
}
