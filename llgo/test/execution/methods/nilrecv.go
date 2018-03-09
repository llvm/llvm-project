// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: true
// CHECK-NEXT: false
// CHECK-NEXT: true
// CHECK-NEXT: false

package main

type T1 int

func (t *T1) t1() { println(t == nil) }

func constNilRecv() {
	(*T1)(nil).t1()
}

func nonConstNilRecv() {
	var v1 T1
	v1.t1()
	var v2 *T1
	v2.t1()
	v2 = &v1
	v2.t1()
}

func main() {
	constNilRecv()
	nonConstNilRecv()
}
