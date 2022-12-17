; REQUIRES: asserts
; RUN: opt -passes=inline -mtriple=aarch64--linux-gnu -mcpu=kryo -S -debug-only=inline-cost < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

define void @outer1(ptr %ptr, i32 %i) {
  call void @inner1(ptr %ptr, i32 %i)
  ret void
}

define void @outer2(ptr %ptr, i32 %i) {
  call void @inner2(ptr %ptr, i32 %i)
  ret void
}

define void @outer3(ptr %ptr, i32 %j) {
  call void @inner3(ptr %ptr, i32 0, i32 %j)
  ret void
}

; The gep in inner1() is reg+reg, which is a legal addressing mode for AArch64.
; Thus, both the gep and ret can be simplified.
; CHECK: Analyzing call of inner1
; CHECK: NumInstructionsSimplified: 2
; CHECK: NumInstructions: 2
define void @inner1(ptr %ptr, i32 %i) {
  %G = getelementptr inbounds [4 x i32], ptr %ptr, i32 0, i32 %i
  ret void
}

; The gep in inner2() is reg+imm+reg, which is not a legal addressing mode for
; AArch64.  Thus, only the ret can be simplified and not the gep.
; CHECK: Analyzing call of inner2
; CHECK: NumInstructionsSimplified: 1
; CHECK: NumInstructions: 2
define void @inner2(ptr %ptr, i32 %i) {
  %G = getelementptr inbounds [4 x i32], ptr %ptr, i32 1, i32 %i
  ret void
}

; The gep in inner3() is reg+reg because %i is a known constant from the
; callsite. This case is a legal addressing mode for AArch64.  Thus, both the
; gep and ret can be simplified.
; CHECK: Analyzing call of inner3
; CHECK: NumInstructionsSimplified: 2
; CHECK: NumInstructions: 2
define void @inner3(ptr %ptr, i32 %i, i32 %j) {
  %G = getelementptr inbounds [4 x i32], ptr %ptr, i32 %i, i32 %j
  ret void
}
