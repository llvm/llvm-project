// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: (+1.000000e+000+2.000000e+000i)
// CHECK-NEXT: (-1.000000e+000-2.000000e+000i)
// CHECK-NEXT: true
// CHECK-NEXT: (+1.000000e+000+2.000000e+000i)
// CHECK-NEXT: (-1.000000e+000-2.000000e+000i)
// CHECK-NEXT: true

package main

func main() {
	var f32 float32 = 1
	var f64 float64 = 1
	c64 := complex(f32, f32+1)
	println(c64)
	println(-c64)
	println(c64 == c64)
	c128 := complex(f64, f64+1)
	println(c128)
	println(-c128)
	println(c128 == c128)
}
