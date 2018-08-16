// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: 0
// CHECK-NEXT: 1
// CHECK-NEXT: 2
// CHECK-NEXT: 0
// CHECK-NEXT: 1
// CHECK-NEXT: 2
// CHECK-NEXT: 3
// CHECK-NEXT: 4
// CHECK-NEXT: 5
// CHECK-NEXT: 0 3
// CHECK-NEXT: 1 4
// CHECK-NEXT: 2 5
// CHECK-NEXT: 1
// CHECK-NEXT: done

package main

func main() {
	defer println("done")
	m := make(map[int]int)
	m[0] = 3
	m[1] = 4
	m[2] = 5
	for k := range m {
		println(k)
	}
	for k, _ := range m {
		println(k)
	}
	for _, v := range m {
		println(v)
	}
	for k, v := range m {
		println(k, v)
	}

	// test deletion.
	i := 0
	for k, _ := range m {
		i++
		delete(m, (k+1)%3)
		delete(m, (k+2)%3)
	}
	println(i)
}
