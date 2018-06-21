// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: 1
// CHECK-NEXT: 2

package main

func main() {
	// case clauses have their own scope.
	switch {
	case true, false:
		x := 1
		println(x)
		fallthrough
	default:
		x := 2
		println(x)
	}
}
