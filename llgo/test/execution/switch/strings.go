// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: abc
// CHECK-NEXT: def, abc

package main

func main() {
	switch "abc" {
	case "def":
		println("def")
	case "abc":
		println("abc")
	}

	switch "abc" {
	case "def", "abc":
		println("def, abc")
	}
}
