; RUN: opt < %s -passes=indvars -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; PR25372
; We can compute the expression of %phi0 and that is a SCEV
; constant. However, instcombine can't deduce this, so we can
; potentially end up trying to handle a constant when replacing
; congruent IVs.

; CHECK-LABEL: crash
define void @crash() {
entry:
  br i1 false, label %not_taken, label %pre

not_taken:
  br label %pre

pre:
; %phi0.pre and %phi1.pre are evaluated by SCEV to constant 0.
  %phi0.pre = phi i32 [ 0, %entry ], [ 2, %not_taken ]
  %phi1.pre = phi i32 [ 0, %entry ], [ 1, %not_taken ]
  br label %loop

loop:
; %phi0 and %phi1 are evaluated by SCEV to constant 0.
  %phi0 = phi i32 [ 0, %loop ], [ %phi0.pre, %pre ]
  %phi1 = phi i32 [ 0, %loop ], [ %phi1.pre, %pre ]
  br i1 undef, label %exit, label %loop

exit:
  ret void
}
