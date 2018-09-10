// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: 123
// CHECK-NEXT: 123
// CHECK-NEXT: 123
// CHECK-NEXT: 123
// CHECK-NEXT: 123
// CHECK-NEXT: 123
// CHECK-NEXT: 123
// CHECK-NEXT: 123
// CHECK-NEXT: -123
// CHECK-NEXT: -123
// CHECK-NEXT: -123
// CHECK-NEXT: -123
// CHECK-NEXT: 133
// CHECK-NEXT: 65413
// CHECK-NEXT: 4294967173
// CHECK-NEXT: 18446744073709551493
// CHECK-NEXT: 123
// CHECK-NEXT: 123
// CHECK-NEXT: 123
// CHECK-NEXT: 123
// CHECK-NEXT: 123
// CHECK-NEXT: 123
// CHECK-NEXT: 123
// CHECK-NEXT: 123
// CHECK-NEXT: -123
// CHECK-NEXT: -123
// CHECK-NEXT: -123
// CHECK-NEXT: -123
// CHECK-NEXT: 133
// CHECK-NEXT: 65413
// CHECK-NEXT: 4294967173
// CHECK-NEXT: 18446744073709551493
// CHECK-NEXT: +1.230000e+002
// CHECK-NEXT: +1.230000e+002
// CHECK-NEXT: +1.230000e+002
// CHECK-NEXT: +1.230000e+002
// CHECK-NEXT: +1.234500e+004
// CHECK-NEXT: +1.234500e+004
// CHECK-NEXT: +1.234500e+004
// CHECK-NEXT: +1.234500e+004
// CHECK-NEXT: +1.234560e+005
// CHECK-NEXT: +1.234560e+005
// CHECK-NEXT: +1.234560e+005
// CHECK-NEXT: +1.234560e+005
// CHECK-NEXT: +1.234568e+010
// CHECK-NEXT: +1.234568e+010
// CHECK-NEXT: +1.234568e+010
// CHECK-NEXT: +1.234568e+010

package main

func main() {
	// float to int
	for _, f32 := range []float32{123.456, -123.456} {
		println(int8(f32))
		println(int16(f32))
		println(int32(f32))
		println(int64(f32))
		println(uint8(f32))
		println(uint16(f32))
		println(uint32(f32))
		println(uint64(f32))
	}
	for _, f64 := range []float64{123.456, -123.456} {
		println(int8(f64))
		println(int16(f64))
		println(int32(f64))
		println(int64(f64))
		println(uint8(f64))
		println(uint16(f64))
		println(uint32(f64))
		println(uint64(f64))
	}

	// int to float
	var i8 int8 = 123
	println(float32(i8))
	println(float64(i8))
	var ui8 uint8 = 123
	println(float32(ui8))
	println(float64(ui8))
	var i16 int32 = 12345
	println(float32(i16))
	println(float64(i16))
	var ui16 uint32 = 12345
	println(float32(ui16))
	println(float64(ui16))
	var i32 int32 = 123456
	println(float32(i32))
	println(float64(i32))
	var ui32 uint32 = 123456
	println(float32(ui32))
	println(float64(ui32))
	var i64 int64 = 12345678910
	println(float32(i64))
	println(float64(i64))
	var ui64 uint64 = 12345678910
	println(float32(ui64))
	println(float64(ui64))
}
