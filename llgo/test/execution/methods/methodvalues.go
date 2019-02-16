// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: 246
// CHECK-NEXT: T2.f()
// CHECK-NEXT: 10
// CHECK-NEXT: abc

package main

type T1 struct {
	value int
}

func (t *T1) f(m int) int {
	return m * t.value
}

func f1() {
	var t T1
	var f func(int) int = t.f
	t.value = 2
	println(f(123))
}

type T2 struct{}

func (T2) f() {
	println("T2.f()")
}

func f2() {
	var f func() = T2{}.f
	f()
}

type T3 complex128

func (t T3) f() int {
	return int(real(t))
}

func f3() {
	var f func() int = T3(10).f
	println(f())
}

type T4 string

func (t T4) f() string {
	return string(t)
}

func f4() {
	var f func() string = T4("abc").f
	println(f())
}

func main() {
	f1()
	f2()
	f3()
	f4()
}
