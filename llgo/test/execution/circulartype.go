// RUN: llgo -o %t %s
// RUN: %t 2>&1 | count 0

package main

type A struct {
	b1, b2 B
}

type B struct {
	a1, a2 *A
}

func main() {
	var a A
	_ = a
}
