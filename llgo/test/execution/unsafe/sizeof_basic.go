// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: 1
// CHECK-NEXT: 8
// CHECK-NEXT: 1
// CHECK-NEXT: 2
// CHECK-NEXT: 4
// CHECK-NEXT: 8
// CHECK-NEXT: 8
// CHECK-NEXT: 1
// CHECK-NEXT: 2
// CHECK-NEXT: 4
// CHECK-NEXT: 8
// CHECK-NEXT: 4
// CHECK-NEXT: 8
// CHECK-NEXT: 8
// CHECK-NEXT: 16
// CHECK-NEXT: 16
// CHECK-NEXT: 8
// CHECK-NEXT: 8
// CHECK-NEXT: 1
// CHECK-NEXT: 8
// CHECK-NEXT: 1
// CHECK-NEXT: 2
// CHECK-NEXT: 4
// CHECK-NEXT: 8
// CHECK-NEXT: 8
// CHECK-NEXT: 1
// CHECK-NEXT: 2
// CHECK-NEXT: 4
// CHECK-NEXT: 8
// CHECK-NEXT: 4
// CHECK-NEXT: 8
// CHECK-NEXT: 8
// CHECK-NEXT: 8
// CHECK-NEXT: 8
// CHECK-NEXT: 8
// CHECK-NEXT: 8

package main

import "unsafe"

func main() {
	var b bool
	var i int
	var i8 int8
	var i16 int16
	var i32 int32
	var i64 int64
	var u uint
	var u8 uint8
	var u16 uint16
	var u32 uint32
	var u64 uint64
	var f32 float32
	var f64 float64
	var c64 complex64
	var c128 complex128
	var s string
	var p unsafe.Pointer
	var up uintptr

	println(unsafe.Sizeof(b))
	println(unsafe.Sizeof(i))
	println(unsafe.Sizeof(i8))
	println(unsafe.Sizeof(i16))
	println(unsafe.Sizeof(i32))
	println(unsafe.Sizeof(i64))
	println(unsafe.Sizeof(u))
	println(unsafe.Sizeof(u8))
	println(unsafe.Sizeof(u16))
	println(unsafe.Sizeof(u32))
	println(unsafe.Sizeof(u64))
	println(unsafe.Sizeof(f32))
	println(unsafe.Sizeof(f64))
	println(unsafe.Sizeof(c64))
	println(unsafe.Sizeof(c128))
	println(unsafe.Sizeof(s))
	println(unsafe.Sizeof(p))
	println(unsafe.Sizeof(up))

	println(unsafe.Alignof(b))
	println(unsafe.Alignof(i))
	println(unsafe.Alignof(i8))
	println(unsafe.Alignof(i16))
	println(unsafe.Alignof(i32))
	println(unsafe.Alignof(i64))
	println(unsafe.Alignof(u))
	println(unsafe.Alignof(u8))
	println(unsafe.Alignof(u16))
	println(unsafe.Alignof(u32))
	println(unsafe.Alignof(u64))
	println(unsafe.Alignof(f32))
	println(unsafe.Alignof(f64))
	println(unsafe.Alignof(c64))
	println(unsafe.Alignof(c128))
	println(unsafe.Alignof(s))
	println(unsafe.Alignof(p))
	println(unsafe.Alignof(up))
}
