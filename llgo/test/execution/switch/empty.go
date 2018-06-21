// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: f was called

package main

func f() int {
	println("f was called")
	return 123
}

func main() {
	switch f() {
	}
}
