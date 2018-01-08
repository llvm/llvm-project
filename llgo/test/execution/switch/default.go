// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: default
// CHECK-NEXT: true

package main

func main() {
	switch true {
	default:
		println("default")
	}

	switch {
	default:
		println("default")
	case true:
		println("true")
	}
}
