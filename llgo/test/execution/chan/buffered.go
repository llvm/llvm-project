// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: 0 0
// CHECK-NEXT: 0 1
// CHECK-NEXT: 10
// CHECK-NEXT: 20
// CHECK-NEXT: 30
// CHECK-NEXT: 40
// CHECK-NEXT: 50
// CHECK-NEXT: 60
// CHECK-NEXT: 70
// CHECK-NEXT: 80
// CHECK-NEXT: 90
// CHECK-NEXT: 100
// CHECK-NEXT: -1

package main

func main() {
	c := make(chan int)
	println(len(c), cap(c))
	c1 := make(chan int, 1)
	println(len(c1), cap(c1))
	f := func() {
		n, ok := <-c
		if ok {
			c1 <- n * 10
		} else {
			c1 <- -1
		}
	}
	for i := 0; i < 10; i++ {
		go f()
		c <- i + 1
		println(<-c1)
	}
	go f()
	close(c)
	println(<-c1)
}
