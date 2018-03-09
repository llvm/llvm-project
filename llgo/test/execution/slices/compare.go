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
	var s []int
	println(s == nil)
	println(s != nil)
	println(nil == s)
	println(nil != s)
	s = make([]int, 0)
	println(s == nil)
	println(s != nil)
	println(nil == s)
	println(nil != s)
}
