// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: true
// CHECK-NEXT: false

package main

func main() {
	switch true {
	default:
		break
		println("default")
	}

	switch true {
	case true:
		println("true")
		fallthrough
	case false:
		println("false")
	}
}
