// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: hello from T 1
// CHECK-NEXT: hello from T 2

package main

type T struct {
	val int
}

func (t T) Hello(done chan bool) {
	println("hello from T", t.val)
	done <- true
}

type I interface {
	Hello(chan bool)
}

func main() {
	done := make(chan bool)

	t := T{1}
	go t.Hello(done)
	<-done

	var i I = T{2}
	go i.Hello(done)
	<-done

	go println("hello builtin")
}
