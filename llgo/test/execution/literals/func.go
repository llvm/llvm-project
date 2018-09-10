// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: true
// CHECK-NEXT: false

package main

func main() {
	f := func(x bool) {
		println(x)
	}
	f(true)
	f(false)
}
