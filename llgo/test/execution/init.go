// RUN: llgo -o %t %s %p/Inputs/init2.go
// RUN: %t 2>&1 | FileCheck %s

package main

// CHECK-DAG: do some other stuff before main
//func init()

// CHECK-DAG: do some stuff before main
func init() {
	println("do some stuff before main")
}

// CHECK: main has been called
func main() {
	println("main has been called")
}
