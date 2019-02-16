// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: f2.1
// CHECK-NEXT: f5
// CHECK-NEXT: recovered no error
// CHECK-NEXT: f5
// CHECK-NEXT: recovered: meep meep
// CHECK-NEXT: 888
// CHECK-NEXT: f5
// CHECK-NEXT: recovered no error
// CHECK-NEXT: f5
// CHECK-NEXT: recovered no error
// CHECK-NEXT: 888
// CHECK-NEXT: 456
// CHECK-NEXT: 999
// CHECK-NEXT: 999
// CHECK-NEXT: 123
// CHECK-NEXT: 999
// CHECK-NEXT: 999
// CHECK-NEXT: 246
// CHECK-NEXT: f2.2
// CHECK-NEXT: f2.3
// CHECK-NEXT: f1.1
// CHECK-NEXT: f1.2
// CHECK-NEXT: recovered: second
// CHECK-NEXT: ahoy

package main

type T struct {
	value int
}

type T1 struct {
	T
}

func (t T) abc() {
	println(t.value)
}

func (t *T) def() {
	println(t.value)
}

func (t *T) ghi(v int) {
	println(v)
}

func printerr(err interface{}) {
	if err != nil {
		println("recovered:", err.(string))
	} else {
		println("recovered no error")
	}
}

func f6() {
	defer func() { printerr(recover()) }()
	defer func() { panic("second") }()
	panic("first")
}

func f5(panic_ bool) {
	var t1 T1
	t1.T.value = 888
	defer t1.abc()
	var f func(int)
	f = func(recursion int) {
		if recursion > 0 {
			f(recursion - 1)
			return
		}
		println("f5")
		printerr(recover())
	}
	defer f(0) // will recover (after f(1))
	defer f(1) // won't recover
	if panic_ {
		panic("meep meep")
	}
}

func f4() {
	var a T = T{999}
	var b *T = &a
	defer a.abc()
	defer a.def()
	defer a.ghi(123)
	defer b.abc()
	defer b.def()
	defer b.ghi(456)
	f5(true)
	f5(false) // verify the recover in f5 works
}

func f3() (a int) {
	defer func() { a *= 2 }()
	f4()
	return 123
}

func f2() {
	defer func() { println("f2.3") }()
	defer func(s string) { println(s) }("f2.2")
	println("f2.1")
	println(f3())
}

func f1() {
	defer func() { println("f1.2") }()
	defer func() { println("f1.1") }()
	f2()
}

func builtins() {
	defer println("ahoy")
}

func main() {
	f1()
	f6()
	builtins()
}
