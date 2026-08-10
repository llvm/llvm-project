; RUN: llc < %s -mcpu=corei7 -mtriple=i686-pc-win32

; Make sure we don't crash on this testcase.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

define void @_ZN6VectorIfE3equIeEEvfRKS_IT_E() nounwind uwtable ssp align 2 {
entry:
  br i1 undef, label %while.end, label %while.body.lr.ph

while.body.lr.ph:                                 ; preds = %entry
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %while.body.lr.ph
  %0 = fptrunc <8 x x86_fp80> undef to <8 x float>
  store <8 x float> %0, ptr undef, align 4
  br label %vector.body

while.end:                                        ; preds = %entry
  ret void
}
