; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux -enable-misched < %s | FileCheck %s
;
; PR14315: misched should not move the physreg copy of %t below the calls.

@.str89 = external unnamed_addr constant [6 x i8], align 1

declare void @init() nounwind

declare void @clock() nounwind

; CHECK: mflr 0
; CHECK: fmr 31, 1
; CHECK: bl init
define double @s332(double %t) nounwind {
entry:
  tail call void @init()
  tail call void @clock() nounwind
  br label %for.cond2

for.cond2:                                        ; preds = %for.body4, %entry
  %i.0 = phi i32 [ %inc, %for.body4 ], [ 0, %entry ]
  %cmp3 = icmp slt i32 undef, 16000
  br i1 %cmp3, label %for.body4, label %L20

for.body4:                                        ; preds = %for.cond2
  %cmp5 = fcmp ogt double undef, %t
  %inc = add nsw i32 %i.0, 1
  br i1 %cmp5, label %L20, label %for.cond2

L20:                                              ; preds = %for.body4, %for.cond2
  %index.0 = phi i32 [ -2, %for.cond2 ], [ %i.0, %for.body4 ]
  %index.d = sitofp i32 %index.0 to double
  %retval = fadd double %t, %index.d
  ret double %retval
}
