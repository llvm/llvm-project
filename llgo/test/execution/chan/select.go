// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: sent a value
// CHECK-NEXT: received 123
// CHECK-NEXT: default

package main

func f1() {
	c := make(chan int, 1)
	for i := 0; i < 3; i++ {
		select {
		case n, _ := <-c:
			println("received", n)
			c = nil
		case c <- 123:
			println("sent a value")
		default:
			println("default")
		}
	}
}

func main() {
	f1()
}
