// RUN: llgo -o %t %s
// RUN: %t 2>&1 | FileCheck %s

// CHECK: 9223372036854775808 -63 false
// CHECK-NEXT: 11529215046068469760 -60 false
// CHECK-NEXT: 14411518807585587200 -57 false
// CHECK-NEXT: 18014398509481984000 -54 false
// CHECK-NEXT: 11258999068426240000 -50 false
// CHECK-NEXT: 14073748835532800000 -47 false
// CHECK-NEXT: 17592186044416000000 -44 false
// CHECK-NEXT: 10995116277760000000 -40 false
// CHECK-NEXT: 0 0
// CHECK-NEXT: 1 0
// CHECK-NEXT: 2 1
// CHECK-NEXT: 3 0
// CHECK-NEXT: 4 2
// CHECK-NEXT: 5 0
// CHECK-NEXT: 6 3
// CHECK-NEXT: 7 0
// CHECK-NEXT: 8 4
// CHECK-NEXT: 9 0
// CHECK-NEXT: 0 1
// CHECK-NEXT: 1 2

package main

// An extFloat represents an extended floating-point number, with more
// precision than a float64. It does not try to save bits: the
// number represented by the structure is mant*(2^exp), with a negative
// sign if neg is true.
type extFloat struct {
	mant uint64
	exp  int
	neg  bool
}

var smallPowersOfTen = [...]extFloat{
	{1 << 63, -63, false},        // 1
	{0xa << 60, -60, false},      // 1e1
	{0x64 << 57, -57, false},     // 1e2
	{0x3e8 << 54, -54, false},    // 1e3
	{0x2710 << 50, -50, false},   // 1e4
	{0x186a0 << 47, -47, false},  // 1e5
	{0xf4240 << 44, -44, false},  // 1e6
	{0x989680 << 40, -40, false}, // 1e7
}

var arrayWithHoles = [10]int{
	2: 1,
	4: 2,
	6: 3,
	8: 4,
}

type namedInt int32

const N0 namedInt = 0
const N1 namedInt = 1

var arrayWithNamedIndices = [...]int{
	N0: 1,
	N1: 2,
}

func main() {
	for i := range smallPowersOfTen {
		s := smallPowersOfTen[i]
		println(s.mant, s.exp, s.neg)
	}

	for i, value := range arrayWithHoles {
		println(i, value)
	}

	for i, value := range arrayWithNamedIndices {
		println(i, value)
	}
}
