// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: 0 46 1
// CHECK-NEXT: 0 46
// CHECK-NEXT: 0 169 1
// CHECK-NEXT: 0 169
// CHECK-NEXT: 0 8364 1
// CHECK-NEXT: 0 8364
// CHECK-NEXT: 0 66560 1
// CHECK-NEXT: 0 66560
// CHECK-NEXT: 0 83 1
// CHECK-NEXT: 1 97 2
// CHECK-NEXT: 2 108 3
// CHECK-NEXT: 3 101 4
// CHECK-NEXT: 4 32 5
// CHECK-NEXT: 5 112 6
// CHECK-NEXT: 6 114 7
// CHECK-NEXT: 7 105 8
// CHECK-NEXT: 8 99 9
// CHECK-NEXT: 9 101 10
// CHECK-NEXT: 10 58 11
// CHECK-NEXT: 11 32 12
// CHECK-NEXT: 12 8364 13
// CHECK-NEXT: 15 48 14
// CHECK-NEXT: 16 46 15
// CHECK-NEXT: 17 57 16
// CHECK-NEXT: 18 57 17
// CHECK-NEXT: 0 83
// CHECK-NEXT: 1 97
// CHECK-NEXT: 2 108
// CHECK-NEXT: 3 101
// CHECK-NEXT: 4 32
// CHECK-NEXT: 5 112
// CHECK-NEXT: 6 114
// CHECK-NEXT: 7 105
// CHECK-NEXT: 8 99
// CHECK-NEXT: 9 101
// CHECK-NEXT: 10 58
// CHECK-NEXT: 11 32
// CHECK-NEXT: 12 8364
// CHECK-NEXT: 15 48
// CHECK-NEXT: 16 46
// CHECK-NEXT: 17 57
// CHECK-NEXT: 18 57

package main

func printchars(s string) {
	var x int
	for i, c := range s {
		// test loop-carried dependence (x++), introducing a Phi node
		x++
		println(i, c, x)
	}

	// now test with plain old assignment
	var i int
	var c rune
	for i, c = range s {
		println(i, c)
		if i == len(s)-1 {
			// test multiple branches to loop header
			continue
		}
	}
}

func main() {
	// 1 bytes
	printchars(".")

	// 2 bytes
	printchars("©")

	// 3 bytes
	printchars("€")

	// 4 bytes
	printchars("𐐀")

	// mixed
	printchars("Sale price: €0.99")

	// TODO add test cases for invalid sequences
}
