; RUN: llvm-mc -triple arm64-apple-darwin -show-encoding < %s | FileCheck %s
.text
; parseOperand check
; CHECK: add sp, sp, #32             ; encoding: [0xff,0x83,0x00,0x91]
    add sp, sp, 32

; Optional shift
; CHECK: adds x3, x4, #1024, lsl #12 ; encoding: [0x83,0x00,0x50,0xb1]
adds x3, x4, 1024, lsl 12

; CHECK-COUNT-2: mov x3, #327680 ; encoding: [0xa3,0x00,0xa0,0xd2]
movz x3, (0x54321) >> 16, lsl 16
movz x3, (0x54321 >> 16), lsl 16

; Optional extend
; CHECK: add sp, x2, x3              ; encoding: [0x5f,0x60,0x23,0x8b]
add sp, x2, x3, uxtx 0

; FP immediates
; CHECK: fmov s1, #0.12500000      ; encoding: [0x01,0x10,0x28,0x1e]
fmov s1, 0.125

; Barrier operand
; CHECK: dmb osh    ; encoding: [0xbf,0x33,0x03,0xd5]
dmb 3

; Prefetch and memory

; Single register inside []
; CHECK: ldnp  w3, w2, [x15, #16]       ; encoding: [0xe3,0x09,0x42,0x28]
ldnp  w3, w2, [x15, 16]

; Memory, two registers inside []
; CHECK: prfm   pstl3strm, [x4, x5, lsl #3] ; encoding: [0x95,0x78,0xa5,0xf8]
prfm  pstl3strm, [x4, x5, lsl 3]
