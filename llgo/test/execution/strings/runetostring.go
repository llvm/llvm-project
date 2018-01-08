// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: test( 46 )
// CHECK-NEXT: .
// CHECK-NEXT: 46
// CHECK-NEXT: 0 46
// CHECK-NEXT: test( 169 )
// CHECK-NEXT: ©
// CHECK-NEXT: 194
// CHECK-NEXT: 169
// CHECK-NEXT: 0 169
// CHECK-NEXT: test( 8364 )
// CHECK-NEXT: €
// CHECK-NEXT: 226
// CHECK-NEXT: 130
// CHECK-NEXT: 172
// CHECK-NEXT: 0 8364
// CHECK-NEXT: test( 66560 )
// CHECK-NEXT: 𐐀
// CHECK-NEXT: 240
// CHECK-NEXT: 144
// CHECK-NEXT: 144
// CHECK-NEXT: 128
// CHECK-NEXT: 0 66560
// CHECK-NEXT: .©€𐐀
// CHECK-NEXT: 4 4 4
// CHECK-NEXT: true
// CHECK-NEXT: true
// CHECK-NEXT: true
// CHECK-NEXT: true
// CHECK-NEXT: true
// CHECK-NEXT: true
// CHECK-NEXT: true
// CHECK-NEXT: true

package main

func test(r rune) {
	println("test(", r, ")")
	s := string(r)
	println(s)
	for i := 0; i < len(s); i++ {
		println(s[i])
	}
	for i, r := range s {
		println(i, r)
	}
}

type namedRune rune

func testslice(r1 []rune) {
	s := string(r1)
	println(s)
	r2 := []rune(s)
	r3 := []namedRune(s)
	println(len(r1), len(r2), len(r3))
	if len(r2) == len(r1) && len(r3) == len(r1) {
		for i := range r2 {
			println(r1[i] == r2[i])
			println(r1[i] == rune(r3[i]))
		}
	}
}

func main() {
	var runes = []rune{'.', '©', '€', '𐐀'}
	test(runes[0])
	test(runes[1])
	test(runes[2])
	test(runes[3])
	testslice(runes)
}
