// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: 1
// CHECK-NEXT: 20
// CHECK-NEXT: extra: 10

package main

func swap(a, b int) (int, int) {
	return b, a
}

func sub(a, b int) int {
	return a - b
}

func printint(a int, extra ...int) {
	println(a)
	for _, b := range extra {
		println("extra:", b)
	}
}

func main() {
	println(sub(swap(1, 2)))
	printint(swap(10, 20))
}
