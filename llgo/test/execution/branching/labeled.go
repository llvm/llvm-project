// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: 0

package main

func labeledBreak() {
	var i int
L:
	for ; i < 10; i++ {
		switch {
		default:
			break L
		}
	}
	println(i)
}

func main() {
	labeledBreak()
}
