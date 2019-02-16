// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: 0 false
// CHECK-NEXT: 1 true
// CHECK-NEXT: 1 true
// CHECK-NEXT: 1 true

package main

func main() {
	m := make(map[int]int)
	v, ok := m[8]
	println(v, ok)
	m[8] = 1
	v, ok = m[8]
	println(v, ok)

	type S struct{ s1, s2 string }
	sm := make(map[S]int)
	sm[S{"ABC", "DEF"}] = 1
	sv, ok := sm[S{string([]byte{65, 66, 67}), string([]byte{68, 69, 70})}]
	println(sv, ok)

	type A [2]string
	am := make(map[A]int)
	am[A{"ABC", "DEF"}] = 1
	av, ok := am[A{string([]byte{65, 66, 67}), string([]byte{68, 69, 70})}]
	println(av, ok)
}
