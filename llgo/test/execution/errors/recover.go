// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: (0x0,0x0)

package main

func main() {
	err := recover()
	println(err)
}
