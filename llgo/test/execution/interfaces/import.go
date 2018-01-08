// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

package main

import "syscall"

type Signal interface {
	Signal()
}

func main() {
	var s Signal = syscall.SIGINT
	// CHECK: ({{.*}},{{.*}})
	println(s)
}
