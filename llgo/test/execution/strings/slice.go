// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: abcdef
// CHECK-NEXT: bcdef
// CHECK-NEXT: abc
// CHECK-NEXT: bcd

package main

func main() {
	s := "abcdef"
	println(s[:])
	println(s[1:])
	println(s[:3])
	println(s[1:4])
}
