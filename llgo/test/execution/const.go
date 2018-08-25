// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: 0
// CHECK-NEXT: 1
// CHECK-NEXT: 1 1
// CHECK-NEXT: 1 1 1 4
// CHECK-NEXT: 2147483647
// CHECK-NEXT: -2147483648
// CHECK-NEXT: 2147483647
// CHECK-NEXT: -127
// CHECK-NEXT: false
// CHECK-NEXT: 10000000000
// CHECK-NEXT: 1
// CHECK-NEXT: 3

package main

import "runtime"

const (
	a = iota * 2
	A = 1
	B
	C
	D = Z + iota
)

const (
	Z    = iota
	Big  = 1<<31 - 1
	Big2 = -2147483648
	Big3 = 2147483647
)

const (
	expbits32   uint = 8
	bias32           = -1<<(expbits32-1) + 1
	darwinAMD64      = runtime.GOOS == "darwin" && runtime.GOARCH == "amd64"
)

func f1() float32 {
	return 0
}

func constArrayLen() {
	a := [...]int{1, 2, 3}
	const x = len(a)
	println(x)
}

func main() {
	println(a)
	println(B)
	println(A, A)
	println(A, B, C, D)
	println(Big)
	println(Big2)
	println(Big3)
	println(bias32)

	// Currently fails, due to difference in C printf and Go's println
	// formatting of the exponent.
	//println(10 * 1e9)
	println(darwinAMD64)

	// Test conversion.
	println(int64(10) * 1e9)

	// Ensure consts work just as well when declared inside a function.
	const (
		x_ = iota
		y_
	)
	println(y_)

	constArrayLen()
}
