// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: expected: y != z

package main

type any interface{}

type Stringer interface {
	String() string
}

type lessThanAWord struct {
	a byte
}

func (l lessThanAWord) String() string {
	return "!"
}

func makeAStringer() Stringer {
	return lessThanAWord{}
}

func main() {
	var x1, x2 int = 1, 2
	var y any = x1
	var z any = x2
	if y != z {
		println("expected: y != z")
	} else {
		println("unexpected: y == z")
	}
	/*
		if y == x1 {
			println("expected: y == x1")
		} else {
			println("unexpected: y == x1")
		}
	*/
	//println(y.(int))
}
