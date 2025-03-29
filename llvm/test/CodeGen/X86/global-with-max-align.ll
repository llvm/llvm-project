; RUN: llc -mtriple=x86_64 < %s | FileCheck %s

; Make sure alignment of 2^32 isn't truncated to zero.

; CHECK: .globl  g1
; CHECK-NEXT: .p2align 32, 0x0
; CHECK: .globl  g2
; CHECK-NEXT: .p2align 32, 0x0
; CHECK: .globl  g3
; CHECK-NEXT: .p2align 32, 0x0

@g1 = global i32 0, align 4294967296
@g2 = global i32 33, align 4294967296
@g3 = constant i32 44, align 4294967296
