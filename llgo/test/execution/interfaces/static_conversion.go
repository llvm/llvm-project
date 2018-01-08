// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: 666
// CHECK-NEXT: 3

package main

type Blah interface{}
type Numbered interface {
	Blah
	Number() int
}

type Beast struct{}

func (b *Beast) Number() int {
	return 666
}

type MagicNumber int

func (m MagicNumber) Number() int {
	return int(m)
}

func main() {
	var b Beast
	var m MagicNumber = 3
	var n Numbered = &b
	println(n.Number())

	n = m
	println(n.Number())
}
