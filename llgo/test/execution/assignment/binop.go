// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: 246
// CHECK-NEXT: 123
// CHECK-NEXT: 124
// CHECK-NEXT: 123

package main

func main() {
	x := 123
	x *= 2
	println(x)
	x /= 2
	println(x)
	x += 1
	println(x)
	x -= 1
	println(x)
}
