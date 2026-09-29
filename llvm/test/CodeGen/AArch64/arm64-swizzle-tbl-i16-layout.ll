; RUN: llc < %s -mtriple=arm64-apple-ios7.0 | FileCheck %s
; rdar://13214163 - Make sure we generate a correct lookup table for the TBL
; instruction when the element size of the vector is not 8 bits. We were
; getting both the endianness wrong and the element indexing wrong.
define <8 x i16> @foo(<8 x i16> %a) nounwind readnone {
; CHECK:	.section	__TEXT,__literal16,16byte_literals
; CHECK:	.p2align	4
; CHECK:lCPI0_0:
; CHECK:	.byte	0                       ; 0x0
; CHECK:	.byte	1                       ; 0x1
; CHECK:	.byte	0                       ; 0x0
; CHECK:	.byte	1                       ; 0x1
; CHECK:	.byte	0                       ; 0x0
; CHECK:	.byte	1                       ; 0x1
; CHECK:	.byte	0                       ; 0x0
; CHECK:	.byte	1                       ; 0x1
; CHECK:	.byte	8                       ; 0x8
; CHECK:	.byte	9                       ; 0x9
; CHECK:	.byte	8                       ; 0x8
; CHECK:	.byte	9                       ; 0x9
; CHECK:	.byte	8                       ; 0x8
; CHECK:	.byte	9                       ; 0x9
; CHECK:	.byte	8                       ; 0x8
; CHECK:	.byte	9                       ; 0x9
; CHECK:	.section __TEXT,__text,regular,pure_instructions
; CHECK:	.globl	_foo
; CHECK:	.p2align	2
; CHECK:_foo:                                   ; @foo
; CHECK:	adrp	[[BASE:x[0-9]+]], lCPI0_0@PAGE
; CHECK:	ldr	q[[REG:[0-9]+]], [[[BASE]], lCPI0_0@PAGEOFF]
; CHECK:	tbl.16b	v0, { v0 }, v[[REG]]
; CHECK:	ret

  %val = shufflevector <8 x i16> %a, <8 x i16> undef, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 4, i32 4, i32 4, i32 4>
  ret <8 x i16> %val
}
