// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: 1
// CHECK-NEXT: 1 1
// CHECK-NEXT: 0 0

package main

func main() {
	m := make(map[int]int)
	delete(m, 0) // no-op
	m[0] = 1
	println(len(m))
	delete(m, 1) // no-op
	println(len(m), m[0])
	delete(m, 0) // delete element in map
	println(len(m), m[0])
}
