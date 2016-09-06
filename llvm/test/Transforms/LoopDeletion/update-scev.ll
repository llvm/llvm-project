; RUN: opt -S -analyze -scalar-evolution -loop-deletion -scalar-evolution < %s | FileCheck %s --check-prefix=SCEV-EXPRS
; RUN: opt -S -loop-deletion < %s | FileCheck %s --check-prefix=IR-AFTER-TRANSFORM
; RUN: opt -S -indvars -loop-deletion -indvars < %s | FileCheck %s --check-prefix=ORIGINAL-CRASH

; Checking for a crash.  Loop-deletion would change the loop
; disposition of an instruction, but not update SCEV.

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

define void @pr27570() {
; IR-AFTER-TRANSFORM-LABEL: @pr27570(
; ORIGINAL-CRASH: @pr27570(
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.cond14, %entry
  %f.0 = phi i32 [ 20, %entry ], [ 0, %for.cond14 ]
  br label %for.body

for.body:                                         ; preds = %for.inc11, %for.cond
; IR-AFTER-TRANSFORM: for.body:
; IR-AFTER-TRANSFORM: %cmp = icmp eq i32 %val, -1
; IR-AFTER-TRANSFORM: %conv7 = zext i1 %cmp to i32
; IR-AFTER-TRANSFORM: for.body6:

; SCEV-EXPRS:  %conv7 = zext i1 %cmp to i32
; SCEV-EXPRS:  %conv7 = zext i1 %cmp to i32
; SCEV-EXPRS-NEXT:  -->  {{.*}} LoopDispositions: { %for.body: Variant, %for.cond: Variant, %for.body6: Invariant }
  %val = phi i32 [ -20, %for.cond ], [ %inc12, %for.inc11 ]
  %g.040 = phi i32 [ -20, %for.cond ], [ %and.lcssa, %for.inc11 ]
  br label %for.body6

for.body6:                                        ; preds = %for.body6, %for.body
  %h.039 = phi i32 [ 1, %for.body ], [ %inc, %for.body6 ]
  %g.138 = phi i32 [ %g.040, %for.body ], [ %and, %for.body6 ]
  %cmp = icmp eq i32 %val, -1
  %conv7 = zext i1 %cmp to i32
  %add.i = add nsw i32 %conv7, %h.039
  %sext = shl i32 %add.i, 24
  %conv8 = ashr exact i32 %sext, 24
  %cmp9 = icmp eq i32 %conv8, %f.0
  %conv10 = zext i1 %cmp9 to i32
  %and = and i32 %conv10, %g.138
  %inc = add i32 %h.039, 1
  br i1 undef, label %for.inc11, label %for.body6

for.inc11:                                        ; preds = %for.body6
  %and.lcssa = phi i32 [ %and, %for.body6 ]
  %inc12 = add nsw i32 %val, 1
  %tobool = icmp eq i32 %inc12, 0
  br i1 %tobool, label %for.cond14, label %for.body

for.cond14:                                       ; preds = %for.cond14, %for.inc11
  br i1 undef, label %for.cond, label %for.cond14
}
