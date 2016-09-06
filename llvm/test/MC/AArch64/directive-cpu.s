// RUN: not llvm-mc -triple aarch64-unknown-none-eabi -filetype asm -o - %s 2>&1 | FileCheck %s

	.cpu generic

	fminnm d0, d0, d1

	.cpu generic+fp

	fminnm d0, d0, d1

	.cpu generic+nofp

	fminnm d0, d0, d1

	.cpu generic+simd

	addp v0.4s, v0.4s, v0.4s

	.cpu generic+nosimd

	addp v0.4s, v0.4s, v0.4s

	.cpu generic+crc

	crc32cx w0, w1, x3

	.cpu generic+nocrc

	crc32cx w0, w1, x3

	.cpu generic+crypto+nocrc

	aesd v0.16b, v2.16b

	.cpu generic+nocrypto+crc

	aesd v0.16b, v2.16b

// NOTE: the errors precede the actual output!  The errors appear in order
// though, so validate by hoisting them to the top and preservering relative
// ordering

// CHECK: error: instruction requires: fp-armv8
// CHECK: 	fminnm d0, d0, d1
// CHECK: 	^

// CHECK: error: instruction requires: neon
// CHECK: 	addp v0.4s, v0.4s, v0.4s
// CHECK: 	^

// CHECK: error: instruction requires: crc
// CHECK: 	crc32cx w0, w1, x3
// CHECK: 	^

// CHECK: error: instruction requires: crypto
// CHECK: 	aesd v0.16b, v2.16b
// CHECK: 	^

// CHECK:	fminnm d0, d0, d1
// CHECK:	fminnm d0, d0, d1
// CHECK:	addp v0.4s, v0.4s, v0.4s
// CHECK:	crc32cx w0, w1, x3
// CHECK:	aesd v0.16b, v2.16b
