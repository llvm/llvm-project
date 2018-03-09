// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: 97 98 99

package main

func main() {
	s := "abc"
	println(s[0], s[1], s[2])
}
