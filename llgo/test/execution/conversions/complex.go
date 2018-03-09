// RUN: llgo -o %t %s
// RUN: %t 2>&1 | count 0

package main

func constIntToComplex() complex128 {
	return 0
}

func main() {
	var c64 complex64
	var c128 complex128
	c128 = complex128(c64)
	c64 = complex64(c128)
}
