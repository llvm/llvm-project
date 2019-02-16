// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: 0x0
// CHECK-NEXT: x is nil
// CHECK-NEXT: y is nil
// CHECK-NEXT: z is nil

package main

func main() {
	var x *int = nil
	println(x)

	if x == nil {
		println("x is nil")
	}

	var y interface{}
	var z interface{} = y
	if y == nil {
		println("y is nil")
	} else {
		println("y is not nil")
	}

	if z == nil {
		println("z is nil")
	} else {
		println("z is not nil")
	}
}
