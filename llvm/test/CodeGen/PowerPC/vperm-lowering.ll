; RUN: llc -verify-machineinstrs -O0 -fast-isel=false -mcpu=ppc64 < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

define <16 x i8> @foo() nounwind ssp {
  %1 = shufflevector <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, <16 x i8> <i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, <16 x i32> <i32 0, i32 5, i32 10, i32 15, i32 20, i32 25, i32 30, i32 3, i32 8, i32 13, i32 18, i32 23, i32 28, i32 1, i32 6, i32 11>
  ret <16 x i8> %1
}

; CHECK: .LCPI0_0:
; CHECK: .byte 0
; CHECK: .byte 5
; CHECK: .byte 10
; CHECK: .byte 15
; CHECK: .byte 20
; CHECK: .byte 25
; CHECK: .byte 30
; CHECK: .byte 3
; CHECK: .byte 8
; CHECK: .byte 13
; CHECK: .byte 18
; CHECK: .byte 23
; CHECK: .byte 28
; CHECK: .byte 1
; CHECK: .byte 6
; CHECK: .byte 11
; CHECK: foo:
; CHECK: addis [[REG1:[0-9]+]], 2, .LCPI0_0@toc@ha
; CHECK: addi [[REG2:[0-9]+]], [[REG1]], .LCPI0_0@toc@l
; CHECK: lvx [[REG3:[0-9]+]], 0, [[REG2]]
