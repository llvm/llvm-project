; RUN: opt < %s -passes=simplifycfg -simplifycfg-require-and-preserve-domtree=1 -S | FileCheck %s
; PR9420

; Note that the crash in PR9420 test is sensitive to the ordering of
; the transformations done by SimplifyCFG, so this test is likely to rot
; quickly.

define noalias i8* @func_29() nounwind {
; CHECK: entry:
; CHECK-NEXT: unreachable
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc38, %entry
  %p_34.addr.0 = phi i16 [ 1, %entry ], [ %conv40, %for.inc38 ]
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc29, %for.cond
  %p_32.addr.0 = phi i1 [ true, %for.cond ], [ true, %for.inc29 ]
  br i1 %p_32.addr.0, label %for.body8, label %for.inc38

for.body8:                                        ; preds = %for.cond1
  unreachable

for.inc29:                                        ; preds = %for.cond17
  br label %for.cond1

for.inc38:                                        ; preds = %for.end32
  %conv40 = add i16 %p_34.addr.0, 1
  br label %for.cond
}
