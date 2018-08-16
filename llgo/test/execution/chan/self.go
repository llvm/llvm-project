// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: 1
// CHECK-NEXT: 2
// CHECK-NEXT: true

package main

func main() {
	ch := make(chan int, uint8(1))

	ch <- 1
	println(<-ch)

	ch <- 2
	x, ok := <-ch
	println(x)
	println(ok)
}
