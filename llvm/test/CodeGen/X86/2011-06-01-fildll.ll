; RUN: llc < %s | FileCheck %s
; ModuleID = '<stdin>'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"
target triple = "i386-apple-macosx10.6.6"

define float @f(i64* nocapture %x) nounwind readonly ssp {
entry:
; CHECK: movl
; CHECK-NOT: movl
  %tmp1 = load i64, i64* %x, align 4
; CHECK: fildll
  %conv = sitofp i64 %tmp1 to float
  %add = fadd float %conv, 1.000000e+00
  ret float %add
}
