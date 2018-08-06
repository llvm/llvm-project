; RUN: llc < %s | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"
target triple = "thumbv7-apple-ios0.0.0"

; Codegen should only compare one bit of the loaded value.
; rdar://10887484

; CHECK-LABEL: foo:
; CHECK: ldrb r[[R0:[0-9]+]], [r0]
; CHECK: lsls r{{[0-9]+}}, r[[R0]], #31
define void @foo(i8* %call, double* %p) nounwind {
entry:
  %tmp2 = load i8, i8* %call
  %tmp3 = trunc i8 %tmp2 to i1
  %cond = select i1 %tmp3, double 2.000000e+00, double 1.000000e+00
  store double %cond, double* %p
  ret void
}
