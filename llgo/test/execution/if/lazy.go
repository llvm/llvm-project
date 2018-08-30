// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: False()
// CHECK-NEXT: False()
// CHECK-NEXT: false
// CHECK-NEXT: False()
// CHECK-NEXT: True()
// CHECK-NEXT: true
// CHECK-NEXT: True()
// CHECK-NEXT: true
// CHECK-NEXT: True()
// CHECK-NEXT: true
// CHECK-NEXT: False()
// CHECK-NEXT: false
// CHECK-NEXT: False()
// CHECK-NEXT: false
// CHECK-NEXT: True()
// CHECK-NEXT: False()
// CHECK-NEXT: false
// CHECK-NEXT: True()
// CHECK-NEXT: True()
// CHECK-NEXT: true

package main

func False() bool {
	println("False()")
	return false
}

func True() bool {
	println("True()")
	return true
}

func main() {
	println(False() || False())
	println(False() || True())
	println(True() || False())
	println(True() || True())
	println(False() && False())
	println(False() && True())
	println(True() && False())
	println(True() && True())
}
