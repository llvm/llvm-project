// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: int64 123
// CHECK-NEXT: default
// CHECK-NEXT: uint8 or int8
// CHECK-NEXT: uint8 or int8
// CHECK-NEXT: N

package main

func test(i interface{}) {
	switch x := i.(type) {
	case int64:
		println("int64", x)
	// FIXME
	//case string:
	//	println("string", x)
	default:
		println("default")
	}
}

type stringer interface {
	String() string
}

func printany(i interface{}) {
	switch v := i.(type) {
	case nil:
		print("nil", v)
	case stringer:
		print(v.String())
	case error:
		print(v.Error())
	case int:
		print(v)
	case string:
		print(v)
	}
}

func multi(i interface{}) {
	switch i.(type) {
	case uint8, int8:
		println("uint8 or int8")
	default:
		println("something else")
	}
}

type N int

func (n N) String() string { return "N" }

func named() {
	var x interface{} = N(123)
	switch x := x.(type) {
	case N:
		// Test for bug: previously, type switch was
		// assigning underlying type of N (int).
		println(x.String())
	}
}

func main() {
	test(int64(123))
	test("abc")
	multi(uint8(123))
	multi(int8(123))
	named()
}
