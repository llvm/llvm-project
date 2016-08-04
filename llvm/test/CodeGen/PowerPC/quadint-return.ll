; REQUIRES: asserts
; RUN: llc -verify-machineinstrs -O0 -debug -o - < %s 2>&1 | FileCheck %s

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define i128 @foo() nounwind {
entry:
  %x = alloca i128, align 16
  store i128 27, i128* %x, align 16
  %0 = load i128, i128* %x, align 16
  ret i128 %0
}

; CHECK: ********** Function: foo
; CHECK: ********** FAST REGISTER ALLOCATION **********
; CHECK: %X3<def> = COPY %vreg
; CHECK-NEXT: %X4<def> = COPY %vreg
; CHECK-NEXT: BLR
