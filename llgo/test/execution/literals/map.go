// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: false
// CHECK-NEXT: 2
// CHECK-NEXT: 1 0 3
// CHECK-NEXT: 0.1
// CHECK-NEXT: 0.2
// CHECK-NEXT: 0.3

package main

func main() {
	type IntMap map[int]int
	m := IntMap{0: 1, 2: 3}
	println(m == nil)
	println(len(m))
	println(m[0], m[1], m[2])

	f32tostr := map[float32]string{0.1: "0.1", 0.2: "0.2", 0.3: "0.3"}
	println(f32tostr[0.1])
	println(f32tostr[0.2])
	println(f32tostr[0.3])
}
