// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: true
// CHECK-NEXT: false
// CHECK-NEXT: true
// CHECK-NEXT: false
// CHECK-NEXT: false
// CHECK-NEXT: true
// CHECK-NEXT: true
// CHECK-NEXT: false
// CHECK-NEXT: false
// CHECK-NEXT: true

package main

type S0 struct{}

type S1 struct {
	a int
}

type S2 struct {
	a, b int
}

func testS0() {
	println(S0{} == S0{})
	println(S0{} != S0{})
}

func testS1() {
	println(S1{1} == S1{1})
	println(S1{1} != S1{1})
	println(S1{1} == S1{2})
	println(S1{1} != S1{2})
}

func testS2() {
	s1 := S2{1, 2}
	s2 := S2{1, 3}
	println(s1 == s1)
	println(s1 == s2)
	println(s1 != s1)
	println(s1 != s2)
}

func main() {
	testS0()
	testS1()
	testS2()
}
