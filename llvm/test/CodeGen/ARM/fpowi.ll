; RUN: llc -mtriple arm-unknown-linux-gnueabi -filetype asm -o - %s | FileCheck %s
; PR1287

; ModuleID = '<stdin>'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "arm-unknown-linux-gnueabi"

declare double @llvm.powi.f64(double, i32)

define double @_ZSt3powdi(double %__x, i32 %__i) {
entry:
  %tmp3 = call double @llvm.powi.f64(double %__x, i32 %__i)
  ret double %tmp3
}

; CHECK: bl __powidf2

