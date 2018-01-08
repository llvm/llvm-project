// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: testBytesConversion: true
// CHECK-NEXT: 97
// CHECK-NEXT: 98
// CHECK-NEXT: 99
// CHECK-NEXT: abc
// CHECK-NEXT: !bc
// CHECK-NEXT: testBytesCopy: true

package main

type namedByte byte

func testBytesConversion() {
	s := "abc"
	b := []byte(s)
	println("testBytesConversion:", s == string(b))
	nb := []namedByte(s)
	for _, v := range nb {
		println(v)
	}
	b[0] = '!'
	println(s)
	s = string(b)
	b[0] = 'a'
	println(s)
}

func testBytesCopy() {
	s := "abc"
	b := make([]byte, len(s))
	copy(b, s)
	println("testBytesCopy:", string(b) == s)
}

func main() {
	testBytesConversion()
	testBytesCopy()
}
