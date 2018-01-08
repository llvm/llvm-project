// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: 3
// CHECK-NEXT: 4

package main

type T1 *T1

func count(t T1) int {
	if t == nil {
		return 1
	}
	return 1 + count(*t)
}

func testSelfPointer() {
	var a T1
	var b T1
	var c T1 = &b
	*c = &a
	println(count(c))
	println(count(&c))
}

func main() {
	testSelfPointer()
}
