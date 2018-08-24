// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: false

package main

func main() {
	a := false
	f := func() {
		make(chan *bool, 1) <- &a
	}
	f()
	println(a)
}
