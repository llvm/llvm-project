// RUN: llvm-mc -triple=arm64 -mattr=+neon,+fullfp16 -show-encoding < %s | FileCheck %s

// Check that the assembler can handle the documented syntax for AArch64


//------------------------------------------------------------------------------
// Element reverse
//------------------------------------------------------------------------------
         rev64 v0.16b, v31.16b
         rev64 v2.8h, v4.8h
         rev64 v6.4s, v8.4s
         rev64 v1.8b, v9.8b
         rev64 v13.4h, v21.4h
         rev64 v4.2s, v0.2s

// CHECK:	rev64	v0.16b, v31.16b         // encoding: [0xe0,0x0b,0x20,0x4e]
// CHECK:	rev64	v2.8h, v4.8h            // encoding: [0x82,0x08,0x60,0x4e]
// CHECK:	rev64	v6.4s, v8.4s            // encoding: [0x06,0x09,0xa0,0x4e]
// CHECK:	rev64	v1.8b, v9.8b            // encoding: [0x21,0x09,0x20,0x0e]
// CHECK:	rev64	v13.4h, v21.4h          // encoding: [0xad,0x0a,0x60,0x0e]
// CHECK:	rev64	v4.2s, v0.2s            // encoding: [0x04,0x08,0xa0,0x0e]

         rev32 v30.16b, v31.16b
         rev32 v4.8h, v7.8h
         rev32 v21.8b, v1.8b
         rev32 v0.4h, v9.4h

// CHECK:	rev32	v30.16b, v31.16b        // encoding: [0xfe,0x0b,0x20,0x6e]
// CHECK:	rev32	v4.8h, v7.8h            // encoding: [0xe4,0x08,0x60,0x6e]
// CHECK:	rev32	v21.8b, v1.8b           // encoding: [0x35,0x08,0x20,0x2e]
// CHECK:	rev32	v0.4h, v9.4h            // encoding: [0x20,0x09,0x60,0x2e]

         rev16 v30.16b, v31.16b
         rev16 v21.8b, v1.8b

// CHECK:	rev16	v30.16b, v31.16b        // encoding: [0xfe,0x1b,0x20,0x4e]
// CHECK:	rev16	v21.8b, v1.8b           // encoding: [0x35,0x18,0x20,0x0e]

//------------------------------------------------------------------------------
// Signed integer pairwise add long
//------------------------------------------------------------------------------

         saddlp v3.8h, v21.16b
         saddlp v8.4h, v5.8b
         saddlp v9.4s, v1.8h
         saddlp v0.2s, v1.4h
         saddlp v12.2d, v4.4s
         saddlp v17.1d, v28.2s

// CHECK:	saddlp	v3.8h, v21.16b          // encoding: [0xa3,0x2a,0x20,0x4e]
// CHECK:	saddlp	v8.4h, v5.8b            // encoding: [0xa8,0x28,0x20,0x0e]
// CHECK:	saddlp	v9.4s, v1.8h            // encoding: [0x29,0x28,0x60,0x4e]
// CHECK:	saddlp	v0.2s, v1.4h            // encoding: [0x20,0x28,0x60,0x0e]
// CHECK:	saddlp	v12.2d, v4.4s           // encoding: [0x8c,0x28,0xa0,0x4e]
// CHECK:	saddlp	v17.1d, v28.2s          // encoding: [0x91,0x2b,0xa0,0x0e]

//------------------------------------------------------------------------------
// Unsigned integer pairwise add long
//------------------------------------------------------------------------------

         uaddlp v3.8h, v21.16b
         uaddlp v8.4h, v5.8b
         uaddlp v9.4s, v1.8h
         uaddlp v0.2s, v1.4h
         uaddlp v12.2d, v4.4s
         uaddlp v17.1d, v28.2s

// CHECK:	uaddlp	v3.8h, v21.16b          // encoding: [0xa3,0x2a,0x20,0x6e]
// CHECK:	uaddlp	v8.4h, v5.8b            // encoding: [0xa8,0x28,0x20,0x2e]
// CHECK:	uaddlp	v9.4s, v1.8h            // encoding: [0x29,0x28,0x60,0x6e]
// CHECK:	uaddlp	v0.2s, v1.4h            // encoding: [0x20,0x28,0x60,0x2e]
// CHECK:	uaddlp	v12.2d, v4.4s           // encoding: [0x8c,0x28,0xa0,0x6e]
// CHECK:	uaddlp	v17.1d, v28.2s          // encoding: [0x91,0x2b,0xa0,0x2e]

//------------------------------------------------------------------------------
// Signed integer pairwise add and accumulate long
//------------------------------------------------------------------------------

         sadalp v3.8h, v21.16b
         sadalp v8.4h, v5.8b
         sadalp v9.4s, v1.8h
         sadalp v0.2s, v1.4h
         sadalp v12.2d, v4.4s
         sadalp v17.1d, v28.2s

// CHECK:	sadalp	v3.8h, v21.16b          // encoding: [0xa3,0x6a,0x20,0x4e]
// CHECK:	sadalp	v8.4h, v5.8b            // encoding: [0xa8,0x68,0x20,0x0e]
// CHECK:	sadalp	v9.4s, v1.8h            // encoding: [0x29,0x68,0x60,0x4e]
// CHECK:	sadalp	v0.2s, v1.4h            // encoding: [0x20,0x68,0x60,0x0e]
// CHECK:	sadalp	v12.2d, v4.4s           // encoding: [0x8c,0x68,0xa0,0x4e]
// CHECK:	sadalp	v17.1d, v28.2s          // encoding: [0x91,0x6b,0xa0,0x0e]

//------------------------------------------------------------------------------
// Unsigned integer pairwise add and accumulate long
//------------------------------------------------------------------------------

         uadalp v3.8h, v21.16b
         uadalp v8.4h, v5.8b
         uadalp v9.4s, v1.8h
         uadalp v0.2s, v1.4h
         uadalp v12.2d, v4.4s
         uadalp v17.1d, v28.2s

// CHECK:	uadalp	v3.8h, v21.16b          // encoding: [0xa3,0x6a,0x20,0x6e]
// CHECK:	uadalp	v8.4h, v5.8b            // encoding: [0xa8,0x68,0x20,0x2e]
// CHECK:	uadalp	v9.4s, v1.8h            // encoding: [0x29,0x68,0x60,0x6e]
// CHECK:	uadalp	v0.2s, v1.4h            // encoding: [0x20,0x68,0x60,0x2e]
// CHECK:	uadalp	v12.2d, v4.4s           // encoding: [0x8c,0x68,0xa0,0x6e]
// CHECK:	uadalp	v17.1d, v28.2s          // encoding: [0x91,0x6b,0xa0,0x2e]

//------------------------------------------------------------------------------
// Signed integer saturating accumulate of unsigned value
//------------------------------------------------------------------------------

         suqadd v0.16b, v31.16b
         suqadd v2.8h, v4.8h
         suqadd v6.4s, v8.4s
         suqadd v6.2d, v8.2d
         suqadd v1.8b, v9.8b
         suqadd v13.4h, v21.4h
         suqadd v4.2s, v0.2s

// CHECK:	suqadd	v0.16b, v31.16b         // encoding: [0xe0,0x3b,0x20,0x4e]
// CHECK:	suqadd	v2.8h, v4.8h            // encoding: [0x82,0x38,0x60,0x4e]
// CHECK:	suqadd	v6.4s, v8.4s            // encoding: [0x06,0x39,0xa0,0x4e]
// CHECK:	suqadd	v6.2d, v8.2d            // encoding: [0x06,0x39,0xe0,0x4e]
// CHECK:	suqadd	v1.8b, v9.8b            // encoding: [0x21,0x39,0x20,0x0e]
// CHECK:	suqadd	v13.4h, v21.4h          // encoding: [0xad,0x3a,0x60,0x0e]
// CHECK:	suqadd	v4.2s, v0.2s            // encoding: [0x04,0x38,0xa0,0x0e]

//------------------------------------------------------------------------------
// Unsigned integer saturating accumulate of signed value
//------------------------------------------------------------------------------

         usqadd v0.16b, v31.16b
         usqadd v2.8h, v4.8h
         usqadd v6.4s, v8.4s
         usqadd v6.2d, v8.2d
         usqadd v1.8b, v9.8b
         usqadd v13.4h, v21.4h
         usqadd v4.2s, v0.2s

// CHECK:	usqadd	v0.16b, v31.16b         // encoding: [0xe0,0x3b,0x20,0x6e]
// CHECK:	usqadd	v2.8h, v4.8h            // encoding: [0x82,0x38,0x60,0x6e]
// CHECK:	usqadd	v6.4s, v8.4s            // encoding: [0x06,0x39,0xa0,0x6e]
// CHECK:	usqadd	v6.2d, v8.2d            // encoding: [0x06,0x39,0xe0,0x6e]
// CHECK:	usqadd	v1.8b, v9.8b            // encoding: [0x21,0x39,0x20,0x2e]
// CHECK:	usqadd	v13.4h, v21.4h          // encoding: [0xad,0x3a,0x60,0x2e]
// CHECK:	usqadd	v4.2s, v0.2s            // encoding: [0x04,0x38,0xa0,0x2e]

//------------------------------------------------------------------------------
// Integer saturating absolute
//------------------------------------------------------------------------------

         sqabs v0.16b, v31.16b
         sqabs v2.8h, v4.8h
         sqabs v6.4s, v8.4s
         sqabs v6.2d, v8.2d
         sqabs v1.8b, v9.8b
         sqabs v13.4h, v21.4h
         sqabs v4.2s, v0.2s

// CHECK:	sqabs	v0.16b, v31.16b         // encoding: [0xe0,0x7b,0x20,0x4e]
// CHECK:	sqabs	v2.8h, v4.8h            // encoding: [0x82,0x78,0x60,0x4e]
// CHECK:	sqabs	v6.4s, v8.4s            // encoding: [0x06,0x79,0xa0,0x4e]
// CHECK:	sqabs	v6.2d, v8.2d            // encoding: [0x06,0x79,0xe0,0x4e]
// CHECK:	sqabs	v1.8b, v9.8b            // encoding: [0x21,0x79,0x20,0x0e]
// CHECK:	sqabs	v13.4h, v21.4h          // encoding: [0xad,0x7a,0x60,0x0e]
// CHECK:	sqabs	v4.2s, v0.2s            // encoding: [0x04,0x78,0xa0,0x0e]

//------------------------------------------------------------------------------
// Signed integer saturating negate
//------------------------------------------------------------------------------

         sqneg v0.16b, v31.16b
         sqneg v2.8h, v4.8h
         sqneg v6.4s, v8.4s
         sqneg v6.2d, v8.2d
         sqneg v1.8b, v9.8b
         sqneg v13.4h, v21.4h
         sqneg v4.2s, v0.2s

// CHECK:	sqneg	v0.16b, v31.16b         // encoding: [0xe0,0x7b,0x20,0x6e]
// CHECK:	sqneg	v2.8h, v4.8h            // encoding: [0x82,0x78,0x60,0x6e]
// CHECK:	sqneg	v6.4s, v8.4s            // encoding: [0x06,0x79,0xa0,0x6e]
// CHECK:	sqneg	v6.2d, v8.2d            // encoding: [0x06,0x79,0xe0,0x6e]
// CHECK:	sqneg	v1.8b, v9.8b            // encoding: [0x21,0x79,0x20,0x2e]
// CHECK:	sqneg	v13.4h, v21.4h          // encoding: [0xad,0x7a,0x60,0x2e]
// CHECK:	sqneg	v4.2s, v0.2s            // encoding: [0x04,0x78,0xa0,0x2e]

//------------------------------------------------------------------------------
// Integer absolute
//------------------------------------------------------------------------------

         abs v0.16b, v31.16b
         abs v2.8h, v4.8h
         abs v6.4s, v8.4s
         abs v6.2d, v8.2d
         abs v1.8b, v9.8b
         abs v13.4h, v21.4h
         abs v4.2s, v0.2s

// CHECK:	abs	v0.16b, v31.16b         // encoding: [0xe0,0xbb,0x20,0x4e]
// CHECK:	abs	v2.8h, v4.8h            // encoding: [0x82,0xb8,0x60,0x4e]
// CHECK:	abs	v6.4s, v8.4s            // encoding: [0x06,0xb9,0xa0,0x4e]
// CHECK:	abs	v6.2d, v8.2d            // encoding: [0x06,0xb9,0xe0,0x4e]
// CHECK:	abs	v1.8b, v9.8b            // encoding: [0x21,0xb9,0x20,0x0e]
// CHECK:	abs	v13.4h, v21.4h          // encoding: [0xad,0xba,0x60,0x0e]
// CHECK:	abs	v4.2s, v0.2s            // encoding: [0x04,0xb8,0xa0,0x0e]

//------------------------------------------------------------------------------
// Integer negate
//------------------------------------------------------------------------------

         neg v0.16b, v31.16b
         neg v2.8h, v4.8h
         neg v6.4s, v8.4s
         neg v6.2d, v8.2d
         neg v1.8b, v9.8b
         neg v13.4h, v21.4h
         neg v4.2s, v0.2s

// CHECK:	neg	v0.16b, v31.16b         // encoding: [0xe0,0xbb,0x20,0x6e]
// CHECK:	neg	v2.8h, v4.8h            // encoding: [0x82,0xb8,0x60,0x6e]
// CHECK:	neg	v6.4s, v8.4s            // encoding: [0x06,0xb9,0xa0,0x6e]
// CHECK:	neg	v6.2d, v8.2d            // encoding: [0x06,0xb9,0xe0,0x6e]
// CHECK:	neg	v1.8b, v9.8b            // encoding: [0x21,0xb9,0x20,0x2e]
// CHECK:	neg	v13.4h, v21.4h          // encoding: [0xad,0xba,0x60,0x2e]
// CHECK:	neg	v4.2s, v0.2s            // encoding: [0x04,0xb8,0xa0,0x2e]

//------------------------------------------------------------------------------
// Integer count leading sign bits
//------------------------------------------------------------------------------

         cls v0.16b, v31.16b
         cls v2.8h, v4.8h
         cls v6.4s, v8.4s
         cls v1.8b, v9.8b
         cls v13.4h, v21.4h
         cls v4.2s, v0.2s

// CHECK:	cls	v0.16b, v31.16b         // encoding: [0xe0,0x4b,0x20,0x4e]
// CHECK:	cls	v2.8h, v4.8h            // encoding: [0x82,0x48,0x60,0x4e]
// CHECK:	cls	v6.4s, v8.4s            // encoding: [0x06,0x49,0xa0,0x4e]
// CHECK:	cls	v1.8b, v9.8b            // encoding: [0x21,0x49,0x20,0x0e]
// CHECK:	cls	v13.4h, v21.4h          // encoding: [0xad,0x4a,0x60,0x0e]
// CHECK:	cls	v4.2s, v0.2s            // encoding: [0x04,0x48,0xa0,0x0e]

//------------------------------------------------------------------------------
// Integer count leading zeros
//------------------------------------------------------------------------------

         clz v0.16b, v31.16b
         clz v2.8h, v4.8h
         clz v6.4s, v8.4s
         clz v1.8b, v9.8b
         clz v13.4h, v21.4h
         clz v4.2s, v0.2s

// CHECK:	clz	v0.16b, v31.16b         // encoding: [0xe0,0x4b,0x20,0x6e]
// CHECK:	clz	v2.8h, v4.8h            // encoding: [0x82,0x48,0x60,0x6e]
// CHECK:	clz	v6.4s, v8.4s            // encoding: [0x06,0x49,0xa0,0x6e]
// CHECK:	clz	v1.8b, v9.8b            // encoding: [0x21,0x49,0x20,0x2e]
// CHECK:	clz	v13.4h, v21.4h          // encoding: [0xad,0x4a,0x60,0x2e]
// CHECK:	clz	v4.2s, v0.2s            // encoding: [0x04,0x48,0xa0,0x2e]

//------------------------------------------------------------------------------
// Population count
//------------------------------------------------------------------------------

         cnt v0.16b, v31.16b
         cnt v1.8b, v9.8b

// CHECK:	cnt	v0.16b, v31.16b         // encoding: [0xe0,0x5b,0x20,0x4e]
// CHECK:	cnt	v1.8b, v9.8b            // encoding: [0x21,0x59,0x20,0x0e]

//------------------------------------------------------------------------------
// Bitwise NOT
//------------------------------------------------------------------------------

         not v0.16b, v31.16b
         not v1.8b, v9.8b

// CHECK: {{mvn|not}} v0.16b, v31.16b         // encoding: [0xe0,0x5b,0x20,0x6e]
// CHECK: {{mvn|not}} v1.8b, v9.8b            // encoding: [0x21,0x59,0x20,0x2e]

//------------------------------------------------------------------------------
// Bitwise reverse
//------------------------------------------------------------------------------

         rbit v0.16b, v31.16b
         rbit v1.8b, v9.8b

// CHECK:	rbit	v0.16b, v31.16b         // encoding: [0xe0,0x5b,0x60,0x6e]
// CHECK:	rbit	v1.8b, v9.8b            // encoding: [0x21,0x59,0x60,0x2e]

//------------------------------------------------------------------------------
// Floating-point absolute
//------------------------------------------------------------------------------

         fabs v4.4h, v0.4h
         fabs v6.8h, v8.8h
         fabs v6.4s, v8.4s
         fabs v6.2d, v8.2d
         fabs v4.2s, v0.2s

// CHECK: fabs    v4.4h, v0.4h            // encoding: [0x04,0xf8,0xf8,0x0e]
// CHECK: fabs    v6.8h, v8.8h            // encoding: [0x06,0xf9,0xf8,0x4e]
// CHECK:	fabs	v6.4s, v8.4s            // encoding: [0x06,0xf9,0xa0,0x4e]
// CHECK:	fabs	v6.2d, v8.2d            // encoding: [0x06,0xf9,0xe0,0x4e]
// CHECK:	fabs	v4.2s, v0.2s            // encoding: [0x04,0xf8,0xa0,0x0e]

//------------------------------------------------------------------------------
// Floating-point negate
//------------------------------------------------------------------------------

         fneg v4.4h, v0.4h
         fneg v6.8h, v8.8h
         fneg v6.4s, v8.4s
         fneg v6.2d, v8.2d
         fneg v4.2s, v0.2s

// CHECK: fneg    v4.4h, v0.4h            // encoding: [0x04,0xf8,0xf8,0x2e]
// CHECK: fneg    v6.8h, v8.8h            // encoding: [0x06,0xf9,0xf8,0x6e]
// CHECK:	fneg	v6.4s, v8.4s            // encoding: [0x06,0xf9,0xa0,0x6e]
// CHECK:	fneg	v6.2d, v8.2d            // encoding: [0x06,0xf9,0xe0,0x6e]
// CHECK:	fneg	v4.2s, v0.2s            // encoding: [0x04,0xf8,0xa0,0x2e]

//------------------------------------------------------------------------------
// Integer extract and narrow
//------------------------------------------------------------------------------

         xtn2 v0.16b, v31.8h
         xtn2 v2.8h, v4.4s
         xtn2 v6.4s, v8.2d
         xtn v1.8b, v9.8h
         xtn v13.4h, v21.4s
         xtn v4.2s, v0.2d

// CHECK:	xtn2	v0.16b, v31.8h          // encoding: [0xe0,0x2b,0x21,0x4e]
// CHECK:	xtn2	v2.8h, v4.4s            // encoding: [0x82,0x28,0x61,0x4e]
// CHECK:	xtn2	v6.4s, v8.2d            // encoding: [0x06,0x29,0xa1,0x4e]
// CHECK:	xtn	v1.8b, v9.8h            // encoding: [0x21,0x29,0x21,0x0e]
// CHECK:	xtn	v13.4h, v21.4s          // encoding: [0xad,0x2a,0x61,0x0e]
// CHECK:	xtn	v4.2s, v0.2d            // encoding: [0x04,0x28,0xa1,0x0e]

//------------------------------------------------------------------------------
// Signed integer saturating extract and unsigned narrow
//------------------------------------------------------------------------------

         sqxtun2 v0.16b, v31.8h
         sqxtun2 v2.8h, v4.4s
         sqxtun2 v6.4s, v8.2d
         sqxtun v1.8b, v9.8h
         sqxtun v13.4h, v21.4s
         sqxtun v4.2s, v0.2d

// CHECK:	sqxtun2	v0.16b, v31.8h          // encoding: [0xe0,0x2b,0x21,0x6e]
// CHECK:	sqxtun2	v2.8h, v4.4s            // encoding: [0x82,0x28,0x61,0x6e]
// CHECK:	sqxtun2	v6.4s, v8.2d            // encoding: [0x06,0x29,0xa1,0x6e]
// CHECK:	sqxtun	v1.8b, v9.8h            // encoding: [0x21,0x29,0x21,0x2e]
// CHECK:	sqxtun	v13.4h, v21.4s          // encoding: [0xad,0x2a,0x61,0x2e]
// CHECK:	sqxtun	v4.2s, v0.2d            // encoding: [0x04,0x28,0xa1,0x2e]

//------------------------------------------------------------------------------
// Signed integer saturating extract and narrow
//------------------------------------------------------------------------------

         sqxtn2 v0.16b, v31.8h
         sqxtn2 v2.8h, v4.4s
         sqxtn2 v6.4s, v8.2d
         sqxtn v1.8b, v9.8h
         sqxtn v13.4h, v21.4s
         sqxtn v4.2s, v0.2d

// CHECK:	sqxtn2	v0.16b, v31.8h          // encoding: [0xe0,0x4b,0x21,0x4e]
// CHECK:	sqxtn2	v2.8h, v4.4s            // encoding: [0x82,0x48,0x61,0x4e]
// CHECK:	sqxtn2	v6.4s, v8.2d            // encoding: [0x06,0x49,0xa1,0x4e]
// CHECK:	sqxtn	v1.8b, v9.8h            // encoding: [0x21,0x49,0x21,0x0e]
// CHECK:	sqxtn	v13.4h, v21.4s          // encoding: [0xad,0x4a,0x61,0x0e]
// CHECK:	sqxtn	v4.2s, v0.2d            // encoding: [0x04,0x48,0xa1,0x0e]

//------------------------------------------------------------------------------
// Unsigned integer saturating extract and narrow
//------------------------------------------------------------------------------

         uqxtn2 v0.16b, v31.8h
         uqxtn2 v2.8h, v4.4s
         uqxtn2 v6.4s, v8.2d
         uqxtn v1.8b, v9.8h
         uqxtn v13.4h, v21.4s
         uqxtn v4.2s, v0.2d

// CHECK:	uqxtn2	v0.16b, v31.8h          // encoding: [0xe0,0x4b,0x21,0x6e]
// CHECK:	uqxtn2	v2.8h, v4.4s            // encoding: [0x82,0x48,0x61,0x6e]
// CHECK:	uqxtn2	v6.4s, v8.2d            // encoding: [0x06,0x49,0xa1,0x6e]
// CHECK:	uqxtn	v1.8b, v9.8h            // encoding: [0x21,0x49,0x21,0x2e]
// CHECK:	uqxtn	v13.4h, v21.4s          // encoding: [0xad,0x4a,0x61,0x2e]
// CHECK:	uqxtn	v4.2s, v0.2d            // encoding: [0x04,0x48,0xa1,0x2e]

//------------------------------------------------------------------------------
// Integer shift left long
//------------------------------------------------------------------------------

         shll2 v2.8h, v4.16b, #8
         shll2 v6.4s, v8.8h, #16
         shll2 v6.2d, v8.4s, #32
         shll v2.8h, v4.8b, #8
         shll v6.4s, v8.4h, #16
         shll v6.2d, v8.2s, #32

// CHECK:	shll2	v2.8h, v4.16b, #8      // encoding: [0x82,0x38,0x21,0x6e]
// CHECK:	shll2	v6.4s, v8.8h, #16      // encoding: [0x06,0x39,0x61,0x6e]
// CHECK:	shll2	v6.2d, v8.4s, #32      // encoding: [0x06,0x39,0xa1,0x6e]
// CHECK:	shll	v2.8h, v4.8b, #8       // encoding: [0x82,0x38,0x21,0x2e]
// CHECK:	shll	v6.4s, v8.4h, #16      // encoding: [0x06,0x39,0x61,0x2e]
// CHECK:	shll	v6.2d, v8.2s, #32      // encoding: [0x06,0x39,0xa1,0x2e]

//------------------------------------------------------------------------------
// Floating-point convert downsize
//------------------------------------------------------------------------------

         fcvtn2 v2.8h, v4.4s
         fcvtn2 v6.4s, v8.2d
         fcvtn v13.4h, v21.4s
         fcvtn v4.2s, v0.2d

// CHECK:	fcvtn2	v2.8h, v4.4s            // encoding: [0x82,0x68,0x21,0x4e]
// CHECK:	fcvtn2	v6.4s, v8.2d            // encoding: [0x06,0x69,0x61,0x4e]
// CHECK:	fcvtn	v13.4h, v21.4s          // encoding: [0xad,0x6a,0x21,0x0e]
// CHECK:	fcvtn	v4.2s, v0.2d            // encoding: [0x04,0x68,0x61,0x0e]

//------------------------------------------------------------------------------
// Floating-point convert downsize with inexact
//------------------------------------------------------------------------------

         fcvtxn2 v6.4s, v8.2d
         fcvtxn v4.2s, v0.2d

// CHECK:	fcvtxn2	v6.4s, v8.2d            // encoding: [0x06,0x69,0x61,0x6e]
// CHECK:	fcvtxn	v4.2s, v0.2d            // encoding: [0x04,0x68,0x61,0x2e]

//------------------------------------------------------------------------------
// Floating-point convert upsize
//------------------------------------------------------------------------------

         fcvtl v9.4s, v1.4h
         fcvtl v0.2d, v1.2s
         fcvtl2 v12.4s, v4.8h
         fcvtl2 v17.2d, v28.4s

// CHECK:	fcvtl	v9.4s, v1.4h            // encoding: [0x29,0x78,0x21,0x0e]
// CHECK:	fcvtl	v0.2d, v1.2s            // encoding: [0x20,0x78,0x61,0x0e]
// CHECK:	fcvtl2	v12.4s, v4.8h           // encoding: [0x8c,0x78,0x21,0x4e]
// CHECK:	fcvtl2	v17.2d, v28.4s          // encoding: [0x91,0x7b,0x61,0x4e]

//------------------------------------------------------------------------------
// Floating-point round to integral
//------------------------------------------------------------------------------

         frintn v4.4h, v0.4h
         frintn v6.8h, v8.8h
         frintn v6.4s, v8.4s
         frintn v6.2d, v8.2d
         frintn v4.2s, v0.2s

// CHECK: frintn  v4.4h, v0.4h            // encoding: [0x04,0x88,0x79,0x0e]
// CHECK: frintn  v6.8h, v8.8h            // encoding: [0x06,0x89,0x79,0x4e]
// CHECK:	frintn	v6.4s, v8.4s            // encoding: [0x06,0x89,0x21,0x4e]
// CHECK:	frintn	v6.2d, v8.2d            // encoding: [0x06,0x89,0x61,0x4e]
// CHECK:	frintn	v4.2s, v0.2s            // encoding: [0x04,0x88,0x21,0x0e]

         frinta v4.4h, v0.4h
         frinta v6.8h, v8.8h
         frinta v6.4s, v8.4s
         frinta v6.2d, v8.2d
         frinta v4.2s, v0.2s

// CHECK: frinta  v4.4h, v0.4h            // encoding: [0x04,0x88,0x79,0x2e]
// CHECK: frinta  v6.8h, v8.8h            // encoding: [0x06,0x89,0x79,0x6e]
// CHECK:	frinta	v6.4s, v8.4s            // encoding: [0x06,0x89,0x21,0x6e]
// CHECK:	frinta	v6.2d, v8.2d            // encoding: [0x06,0x89,0x61,0x6e]
// CHECK:	frinta	v4.2s, v0.2s            // encoding: [0x04,0x88,0x21,0x2e]

         frintp v4.4h, v0.4h
         frintp v6.8h, v8.8h
         frintp v6.4s, v8.4s
         frintp v6.2d, v8.2d
         frintp v4.2s, v0.2s

// CHECK: frintp  v4.4h, v0.4h            // encoding: [0x04,0x88,0xf9,0x0e]
// CHECK: frintp  v6.8h, v8.8h            // encoding: [0x06,0x89,0xf9,0x4e]
// CHECK:	frintp	v6.4s, v8.4s            // encoding: [0x06,0x89,0xa1,0x4e]
// CHECK:	frintp	v6.2d, v8.2d            // encoding: [0x06,0x89,0xe1,0x4e]
// CHECK:	frintp	v4.2s, v0.2s            // encoding: [0x04,0x88,0xa1,0x0e]

         frintm v4.4h, v0.4h
         frintm v6.8h, v8.8h
         frintm v6.4s, v8.4s
         frintm v6.2d, v8.2d
         frintm v4.2s, v0.2s

// CHECK: frintm  v4.4h, v0.4h            // encoding: [0x04,0x98,0x79,0x0e]
// CHECK: frintm  v6.8h, v8.8h            // encoding: [0x06,0x99,0x79,0x4e]
// CHECK:	frintm	v6.4s, v8.4s            // encoding: [0x06,0x99,0x21,0x4e]
// CHECK:	frintm	v6.2d, v8.2d            // encoding: [0x06,0x99,0x61,0x4e]
// CHECK:	frintm	v4.2s, v0.2s            // encoding: [0x04,0x98,0x21,0x0e]

         frintx v4.4h, v0.4h
         frintx v6.8h, v8.8h
         frintx v6.4s, v8.4s
         frintx v6.2d, v8.2d
         frintx v4.2s, v0.2s

// CHECK: frintx  v4.4h, v0.4h            // encoding: [0x04,0x98,0x79,0x2e]
// CHECK: frintx  v6.8h, v8.8h            // encoding: [0x06,0x99,0x79,0x6e]
// CHECK:	frintx	v6.4s, v8.4s            // encoding: [0x06,0x99,0x21,0x6e]
// CHECK:	frintx	v6.2d, v8.2d            // encoding: [0x06,0x99,0x61,0x6e]
// CHECK:	frintx	v4.2s, v0.2s            // encoding: [0x04,0x98,0x21,0x2e]

         frintz v4.4h, v0.4h
         frintz v6.8h, v8.8h
         frintz v6.4s, v8.4s
         frintz v6.2d, v8.2d
         frintz v4.2s, v0.2s

// CHECK: frintz  v4.4h, v0.4h            // encoding: [0x04,0x98,0xf9,0x0e]
// CHECK: frintz  v6.8h, v8.8h            // encoding: [0x06,0x99,0xf9,0x4e]
// CHECK:	frintz	v6.4s, v8.4s            // encoding: [0x06,0x99,0xa1,0x4e]
// CHECK:	frintz	v6.2d, v8.2d            // encoding: [0x06,0x99,0xe1,0x4e]
// CHECK:	frintz	v4.2s, v0.2s            // encoding: [0x04,0x98,0xa1,0x0e]

         frinti v4.4h, v0.4h
         frinti v6.8h, v8.8h
         frinti v6.4s, v8.4s
         frinti v6.2d, v8.2d
         frinti v4.2s, v0.2s

// CHECK: frinti  v4.4h, v0.4h            // encoding: [0x04,0x98,0xf9,0x2e]
// CHECK: frinti  v6.8h, v8.8h            // encoding: [0x06,0x99,0xf9,0x6e]
// CHECK:	frinti	v6.4s, v8.4s            // encoding: [0x06,0x99,0xa1,0x6e]
// CHECK:	frinti	v6.2d, v8.2d            // encoding: [0x06,0x99,0xe1,0x6e]
// CHECK:	frinti	v4.2s, v0.2s            // encoding: [0x04,0x98,0xa1,0x2e]

//------------------------------------------------------------------------------
// Floating-point convert to integer
//------------------------------------------------------------------------------

         fcvtns v4.4h, v0.4h
         fcvtns v6.8h, v8.8h
         fcvtns v6.4s, v8.4s
         fcvtns v6.2d, v8.2d
         fcvtns v4.2s, v0.2s

// CHECK: fcvtns  v4.4h, v0.4h            // encoding: [0x04,0xa8,0x79,0x0e]
// CHECK: fcvtns  v6.8h, v8.8h            // encoding: [0x06,0xa9,0x79,0x4e]
// CHECK:	fcvtns	v6.4s, v8.4s            // encoding: [0x06,0xa9,0x21,0x4e]
// CHECK:	fcvtns	v6.2d, v8.2d            // encoding: [0x06,0xa9,0x61,0x4e]
// CHECK:	fcvtns	v4.2s, v0.2s            // encoding: [0x04,0xa8,0x21,0x0e]

         fcvtnu v4.4h, v0.4h
         fcvtnu v6.8h, v8.8h
         fcvtnu v6.4s, v8.4s
         fcvtnu v6.2d, v8.2d
         fcvtnu v4.2s, v0.2s

// CHECK: fcvtnu  v4.4h, v0.4h            // encoding: [0x04,0xa8,0x79,0x2e]
// CHECK: fcvtnu  v6.8h, v8.8h            // encoding: [0x06,0xa9,0x79,0x6e]
// CHECK:	fcvtnu	v6.4s, v8.4s            // encoding: [0x06,0xa9,0x21,0x6e]
// CHECK:	fcvtnu	v6.2d, v8.2d            // encoding: [0x06,0xa9,0x61,0x6e]
// CHECK:	fcvtnu	v4.2s, v0.2s            // encoding: [0x04,0xa8,0x21,0x2e]

         fcvtps v4.4h, v0.4h
         fcvtps v6.8h, v8.8h
         fcvtps v6.4s, v8.4s
         fcvtps v6.2d, v8.2d
         fcvtps v4.2s, v0.2s

// CHECK: fcvtps  v4.4h, v0.4h            // encoding: [0x04,0xa8,0xf9,0x0e]
// CHECK: fcvtps  v6.8h, v8.8h            // encoding: [0x06,0xa9,0xf9,0x4e]
// CHECK:	fcvtps	v6.4s, v8.4s            // encoding: [0x06,0xa9,0xa1,0x4e]
// CHECK:	fcvtps	v6.2d, v8.2d            // encoding: [0x06,0xa9,0xe1,0x4e]
// CHECK:	fcvtps	v4.2s, v0.2s            // encoding: [0x04,0xa8,0xa1,0x0e]

         fcvtpu v4.4h, v0.4h
         fcvtpu v6.8h, v8.8h
         fcvtpu v6.4s, v8.4s
         fcvtpu v6.2d, v8.2d
         fcvtpu v4.2s, v0.2s

// CHECK: fcvtpu  v4.4h, v0.4h            // encoding: [0x04,0xa8,0xf9,0x2e]
// CHECK: fcvtpu  v6.8h, v8.8h            // encoding: [0x06,0xa9,0xf9,0x6e]
// CHECK:	fcvtpu	v6.4s, v8.4s            // encoding: [0x06,0xa9,0xa1,0x6e]
// CHECK:	fcvtpu	v6.2d, v8.2d            // encoding: [0x06,0xa9,0xe1,0x6e]
// CHECK:	fcvtpu	v4.2s, v0.2s            // encoding: [0x04,0xa8,0xa1,0x2e]

         fcvtms v4.4h, v0.4h
         fcvtms v6.8h, v8.8h
         fcvtms v6.4s, v8.4s
         fcvtms v6.2d, v8.2d
         fcvtms v4.2s, v0.2s

// CHECK: fcvtms  v4.4h, v0.4h            // encoding: [0x04,0xb8,0x79,0x0e]
// CHECK: fcvtms  v6.8h, v8.8h            // encoding: [0x06,0xb9,0x79,0x4e]
// CHECK:	fcvtms	v6.4s, v8.4s            // encoding: [0x06,0xb9,0x21,0x4e]
// CHECK:	fcvtms	v6.2d, v8.2d            // encoding: [0x06,0xb9,0x61,0x4e]
// CHECK:	fcvtms	v4.2s, v0.2s            // encoding: [0x04,0xb8,0x21,0x0e]

         fcvtmu v4.4h, v0.4h
         fcvtmu v6.8h, v8.8h
         fcvtmu v6.4s, v8.4s
         fcvtmu v6.2d, v8.2d
         fcvtmu v4.2s, v0.2s

// CHECK: fcvtmu  v4.4h, v0.4h            // encoding: [0x04,0xb8,0x79,0x2e]
// CHECK: fcvtmu  v6.8h, v8.8h            // encoding: [0x06,0xb9,0x79,0x6e]
// CHECK:	fcvtmu	v6.4s, v8.4s            // encoding: [0x06,0xb9,0x21,0x6e]
// CHECK:	fcvtmu	v6.2d, v8.2d            // encoding: [0x06,0xb9,0x61,0x6e]
// CHECK:	fcvtmu	v4.2s, v0.2s            // encoding: [0x04,0xb8,0x21,0x2e]

         fcvtzs v4.4h, v0.4h
         fcvtzs v6.8h, v8.8h
         fcvtzs v6.4s, v8.4s
         fcvtzs v6.2d, v8.2d
         fcvtzs v4.2s, v0.2s

// CHECK: fcvtzs  v4.4h, v0.4h            // encoding: [0x04,0xb8,0xf9,0x0e]
// CHECK: fcvtzs  v6.8h, v8.8h            // encoding: [0x06,0xb9,0xf9,0x4e]
// CHECK:	fcvtzs	v6.4s, v8.4s            // encoding: [0x06,0xb9,0xa1,0x4e]
// CHECK:	fcvtzs	v6.2d, v8.2d            // encoding: [0x06,0xb9,0xe1,0x4e]
// CHECK:	fcvtzs	v4.2s, v0.2s            // encoding: [0x04,0xb8,0xa1,0x0e]


         fcvtzu v4.4h, v0.4h
         fcvtzu v6.8h, v8.8h
         fcvtzu v6.4s, v8.4s
         fcvtzu v6.2d, v8.2d
         fcvtzu v4.2s, v0.2s

// CHECK: fcvtzu  v4.4h, v0.4h            // encoding: [0x04,0xb8,0xf9,0x2e]
// CHECK: fcvtzu  v6.8h, v8.8h            // encoding: [0x06,0xb9,0xf9,0x6e]
// CHECK:	fcvtzu	v6.4s, v8.4s            // encoding: [0x06,0xb9,0xa1,0x6e]
// CHECK:	fcvtzu	v6.2d, v8.2d            // encoding: [0x06,0xb9,0xe1,0x6e]
// CHECK:	fcvtzu	v4.2s, v0.2s            // encoding: [0x04,0xb8,0xa1,0x2e]

         fcvtas v4.4h, v0.4h
         fcvtas v6.8h, v8.8h
         fcvtas v6.4s, v8.4s
         fcvtas v6.2d, v8.2d
         fcvtas v4.2s, v0.2s

// CHECK: fcvtas  v4.4h, v0.4h            // encoding: [0x04,0xc8,0x79,0x0e]
// CHECK: fcvtas  v6.8h, v8.8h            // encoding: [0x06,0xc9,0x79,0x4e]
// CHECK:	fcvtas	v6.4s, v8.4s            // encoding: [0x06,0xc9,0x21,0x4e]
// CHECK:	fcvtas	v6.2d, v8.2d            // encoding: [0x06,0xc9,0x61,0x4e]
// CHECK:	fcvtas	v4.2s, v0.2s            // encoding: [0x04,0xc8,0x21,0x0e]

         fcvtau v4.4h, v0.4h
         fcvtau v6.8h, v8.8h
         fcvtau v6.4s, v8.4s
         fcvtau v6.2d, v8.2d
         fcvtau v4.2s, v0.2s

// CHECK: fcvtau  v4.4h, v0.4h            // encoding: [0x04,0xc8,0x79,0x2e]
// CHECK: fcvtau  v6.8h, v8.8h            // encoding: [0x06,0xc9,0x79,0x6e]
// CHECK:	fcvtau	v6.4s, v8.4s            // encoding: [0x06,0xc9,0x21,0x6e]
// CHECK:	fcvtau	v6.2d, v8.2d            // encoding: [0x06,0xc9,0x61,0x6e]
// CHECK:	fcvtau	v4.2s, v0.2s            // encoding: [0x04,0xc8,0x21,0x2e]

         urecpe v6.4s, v8.4s
         urecpe v4.2s, v0.2s

// CHECK:	urecpe	v6.4s, v8.4s            // encoding: [0x06,0xc9,0xa1,0x4e]
// CHECK:	urecpe	v4.2s, v0.2s            // encoding: [0x04,0xc8,0xa1,0x0e]

         ursqrte v6.4s, v8.4s
         ursqrte v4.2s, v0.2s

// CHECK:	ursqrte	v6.4s, v8.4s            // encoding: [0x06,0xc9,0xa1,0x6e]
// CHECK:	ursqrte	v4.2s, v0.2s            // encoding: [0x04,0xc8,0xa1,0x2e]

         scvtf v4.4h, v0.4h
         scvtf v6.8h, v8.8h
         scvtf v6.4s, v8.4s
         scvtf v6.2d, v8.2d
         scvtf v4.2s, v0.2s

// CHECK: scvtf   v4.4h, v0.4h            // encoding: [0x04,0xd8,0x79,0x0e]
// CHECK: scvtf   v6.8h, v8.8h            // encoding: [0x06,0xd9,0x79,0x4e]
// CHECK:	scvtf	v6.4s, v8.4s            // encoding: [0x06,0xd9,0x21,0x4e]
// CHECK:	scvtf	v6.2d, v8.2d            // encoding: [0x06,0xd9,0x61,0x4e]
// CHECK:	scvtf	v4.2s, v0.2s            // encoding: [0x04,0xd8,0x21,0x0e]

         ucvtf v4.4h, v0.4h
         ucvtf v6.8h, v8.8h
         ucvtf v6.4s, v8.4s
         ucvtf v6.2d, v8.2d
         ucvtf v4.2s, v0.2s

// CHECK: ucvtf   v4.4h, v0.4h            // encoding: [0x04,0xd8,0x79,0x2e]
// CHECK: ucvtf   v6.8h, v8.8h            // encoding: [0x06,0xd9,0x79,0x6e]
// CHECK:	ucvtf	v6.4s, v8.4s            // encoding: [0x06,0xd9,0x21,0x6e]
// CHECK:	ucvtf	v6.2d, v8.2d            // encoding: [0x06,0xd9,0x61,0x6e]
// CHECK:	ucvtf	v4.2s, v0.2s            // encoding: [0x04,0xd8,0x21,0x2e]

         frecpe v4.4h, v0.4h
         frecpe v6.8h, v8.8h
         frecpe v6.4s, v8.4s
         frecpe v6.2d, v8.2d
         frecpe v4.2s, v0.2s

// CHECK: frecpe  v4.4h, v0.4h            // encoding: [0x04,0xd8,0xf9,0x0e]
// CHECK: frecpe  v6.8h, v8.8h            // encoding: [0x06,0xd9,0xf9,0x4e]
// CHECK:	frecpe	v6.4s, v8.4s            // encoding: [0x06,0xd9,0xa1,0x4e]
// CHECK:	frecpe	v6.2d, v8.2d            // encoding: [0x06,0xd9,0xe1,0x4e]
// CHECK:	frecpe	v4.2s, v0.2s            // encoding: [0x04,0xd8,0xa1,0x0e]

         frsqrte v4.4h, v0.4h
         frsqrte v6.8h, v8.8h
         frsqrte v6.4s, v8.4s
         frsqrte v6.2d, v8.2d
         frsqrte v4.2s, v0.2s

// CHECK: frsqrte v4.4h, v0.4h            // encoding: [0x04,0xd8,0xf9,0x2e]
// CHECK: frsqrte v6.8h, v8.8h            // encoding: [0x06,0xd9,0xf9,0x6e]
// CHECK:	frsqrte	v6.4s, v8.4s            // encoding: [0x06,0xd9,0xa1,0x6e]
// CHECK:	frsqrte	v6.2d, v8.2d            // encoding: [0x06,0xd9,0xe1,0x6e]
// CHECK:	frsqrte	v4.2s, v0.2s            // encoding: [0x04,0xd8,0xa1,0x2e]

         fsqrt v4.4h, v0.4h
         fsqrt v6.8h, v8.8h
         fsqrt v6.4s, v8.4s
         fsqrt v6.2d, v8.2d
         fsqrt v4.2s, v0.2s

// CHECK: fsqrt   v4.4h, v0.4h            // encoding: [0x04,0xf8,0xf9,0x2e]
// CHECK: fsqrt   v6.8h, v8.8h            // encoding: [0x06,0xf9,0xf9,0x6e]
// CHECK:	fsqrt	v6.4s, v8.4s            // encoding: [0x06,0xf9,0xa1,0x6e]
// CHECK:	fsqrt	v6.2d, v8.2d            // encoding: [0x06,0xf9,0xe1,0x6e]
// CHECK:	fsqrt	v4.2s, v0.2s            // encoding: [0x04,0xf8,0xa1,0x2e]


