; RUN: llc -mtriple=riscv64 -mattr=+v -verify-machineinstrs < %s | FileCheck %s

define void @constantpool_v16xi8(ptr %x) {
; CHECK: .section	.rodata.cst16,"aM",@progbits,16
; CHECK: .p2align	4, 0x0
; CHECK: .byte
; CHECK: .globl	constantpool_v16xi8
; CHECK: .p2align	2
  store <16 x i8> <i8 0, i8 1, i8 3, i8 3, i8 4, i8 5, i8 15, i8 7, i8 27, i8 9, i8 10, i8 11, i8 12, i8 13, i8 12, i8 15>, ptr %x
  ret void
}

define void @constantpool_v4xi32(ptr %x) {
; CHECK: .section	.rodata.cst16,"aM",@progbits,16
; CHECK: .p2align	4, 0x0
; CHECK: .word
; CHECK: .globl	constantpool_v4xi32
; CHECK: .p2align	2
  store <4 x i32> <i32 -27, i32 255, i32 3, i32 63>, ptr %x
  ret void
}

; Note that to exercise the 64 bit alignment case, we need four elements
; as all of the two element small constant cases get optimized to some
; other sequence
define void @constantpool_v4xi64(ptr %x) {
; CHECK: .section	.rodata.cst32,"aM",@progbits,32
; CHECK: .p2align	5, 0x0
; CHECK: .quad
; CHECK: .globl	constantpool_v4xi64
; CHECK: .p2align	2
  store <4 x i64> <i64 -27, i64 255, i64 3, i64 63>, ptr %x
  ret void
}

define void @constantpool_i64(ptr %x) {
; CHECK: .section	.sdata,"aw",@progbits
; CHECK: .p2align	3
; CHECK: .quad	58373358938439
; CHECK: .globl	constantpool_i64
; CHECK: .p2align	2
  store i64 58373358938439, ptr %x
  ret void
}


