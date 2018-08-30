// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: true
// CHECK-NEXT: false
// CHECK-NEXT: true
// CHECK-NEXT: false
// CHECK-NEXT: false
// CHECK-NEXT: true
// CHECK-NEXT: false
// CHECK-NEXT: true

package main

func main() {
	var f func()
	println(f == nil)
	println(f != nil)
	println(nil == f)
	println(nil != f)
	f = func() {}
	println(f == nil)
	println(f != nil)
	println(nil == f)
	println(nil != f)
}
