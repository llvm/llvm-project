; RUN: opt -S -analyze -scalar-evolution -loop-deletion -scalar-evolution -verify-scev < %s -enable-new-pm=0 | FileCheck %s --check-prefix=SCEV-EXPRS
; RUN: opt -S -passes='print<scalar-evolution>,loop-deletion,print<scalar-evolution>' -verify-scev < %s 2>&1 | FileCheck %s --check-prefix=SCEV-EXPRS
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
  %exit = icmp eq i32 %inc, 20000
  br i1 %exit, label %for.inc11, label %for.body6

for.inc11:                                        ; preds = %for.body6
  %and.lcssa = phi i32 [ %and, %for.body6 ]
  call void @sideeffect(i32 %and.lcssa)
  %inc12 = add nsw i32 %val, 1
  %tobool = icmp eq i32 %inc12, 0
  br i1 %tobool, label %for.cond14, label %for.body

for.cond14:                                       ; preds = %for.cond14, %for.inc11
  br i1 undef, label %for.cond, label %for.cond14
}

declare void @sideeffect(i32)

; LoopDeletion removes the loop %for.body7.1. Make sure %inc.lcssa.1 in the loop
; exit block is correctly invalidated.

define void @test2(double* %bx, i64 %by) local_unnamed_addr align 2 {
; IR-AFTER-TRANSFORM-LABEL: @test2(
; IR-AFTER-TRANSFORM-NOT: for.body7.1:

; SCEV-EXPRS-LABEL: test2
; SCEV-EXPRS:     %inc.lcssa.1 = phi i64 [ undef, %for.body7.preheader.1 ]
; SCEV-EXPRS-NEXT: -->  undef
entry:
  %cmp = icmp sgt i64 %by, 0
  br label %for.cond.preheader

for.cond.preheader:                               ; preds = %entry
  br i1 %cmp, label %for.cond5.preheader.lr.ph, label %for.end14

for.cond5.preheader.lr.ph:                        ; preds = %for.cond.preheader
  br label %for.cond5.preheader

for.cond.loopexit.loopexit:                       ; preds = %for.body10
  %inc11.lcssa = phi i64 [ %inc11, %for.body10 ]
  br label %for.cond.loopexit

for.cond.loopexit:                                ; preds = %for.cond8.preheader, %for.cond.loopexit.loopexit
  %ca.3.lcssa = phi i64 [ %ca.2.lcssa, %for.cond8.preheader ], [ %inc11.lcssa, %for.cond.loopexit.loopexit ]
  br i1 %cmp, label %for.cond5.preheader, label %for.end14.loopexit

for.cond5.preheader:                              ; preds = %for.cond.loopexit, %for.cond5.preheader.lr.ph
  %ca.19 = phi i64 [ 0, %for.cond5.preheader.lr.ph ], [ %ca.3.lcssa, %for.cond.loopexit ]
  br i1 false, label %for.cond8.preheader, label %for.body7.preheader

for.body7.preheader:                              ; preds = %for.cond5.preheader
  br label %for.body7

for.cond8.preheader.loopexit:                     ; preds = %for.body7
  %inc.lcssa = phi i64 [ %inc, %for.body7 ]
  br label %for.cond8.preheader

for.cond8.preheader:                              ; preds = %for.cond8.preheader.loopexit, %for.cond5.preheader
  %ca.2.lcssa = phi i64 [ %ca.19, %for.cond5.preheader ], [ %inc.lcssa, %for.cond8.preheader.loopexit ]
  br i1 true, label %for.body10.preheader, label %for.cond.loopexit

for.body10.preheader:                             ; preds = %for.cond8.preheader
  br label %for.body10

for.body7:                                        ; preds = %for.body7, %for.body7.preheader
  %ca.26 = phi i64 [ %inc, %for.body7 ], [ %ca.19, %for.body7.preheader ]
  %inc = add nsw i64 %ca.26, 1
  %arrayidx = getelementptr inbounds double, double* %bx, i64 %ca.26
  store double 0.000000e+00, double* %arrayidx, align 8
  br i1 false, label %for.cond8.preheader.loopexit, label %for.body7

for.body10:                                       ; preds = %for.body10, %for.body10.preheader
  %ca.37 = phi i64 [ %inc11, %for.body10 ], [ %ca.2.lcssa, %for.body10.preheader ]
  %inc11 = add nsw i64 %ca.37, 1
  br i1 true, label %for.body10, label %for.cond.loopexit.loopexit

for.end14.loopexit:                               ; preds = %for.cond.loopexit
  br label %for.end14

for.end14:                                        ; preds = %for.end14.loopexit, %for.cond.preheader
  br i1 %cmp, label %for.cond5.preheader.lr.ph.1, label %for.end14.1

for.cond5.preheader.lr.ph.1:                      ; preds = %for.end14
  br label %for.cond5.preheader.1

for.cond5.preheader.1:                            ; preds = %for.cond.loopexit.1, %for.cond5.preheader.lr.ph.1
  %ca.19.1 = phi i64 [ 0, %for.cond5.preheader.lr.ph.1 ], [ %ca.3.lcssa.1, %for.cond.loopexit.1 ]
  br i1 true, label %for.cond8.preheader.1, label %for.body7.preheader.1

for.body7.preheader.1:                            ; preds = %for.cond5.preheader.1
  br label %for.body7.1

for.body7.1:                                      ; preds = %for.body7.1, %for.body7.preheader.1
  %ca.26.1 = phi i64 [ %inc.1, %for.body7.1 ], [ %ca.19.1, %for.body7.preheader.1 ]
  %inc.1 = add nsw i64 %ca.26.1, 1
  %arrayidx.1 = getelementptr inbounds double, double* %bx, i64 %ca.26.1
  store double 0.000000e+00, double* %arrayidx.1, align 8
  br i1 true, label %for.cond8.preheader.loopexit.1, label %for.body7.1

for.cond8.preheader.loopexit.1:                   ; preds = %for.body7.1
  %inc.lcssa.1 = phi i64 [ %inc.1, %for.body7.1 ]
  br label %for.cond8.preheader.1

for.cond8.preheader.1:                            ; preds = %for.cond8.preheader.loopexit.1, %for.cond5.preheader.1
  %ca.2.lcssa.1 = phi i64 [ %ca.19.1, %for.cond5.preheader.1 ], [ %inc.lcssa.1, %for.cond8.preheader.loopexit.1 ]
  br i1 false, label %for.body10.preheader.1, label %for.cond.loopexit.1

for.body10.preheader.1:                           ; preds = %for.cond8.preheader.1
  br label %for.body10.1

for.body10.1:                                     ; preds = %for.body10.1, %for.body10.preheader.1
  %ca.37.1 = phi i64 [ %inc11.1, %for.body10.1 ], [ %ca.2.lcssa.1, %for.body10.preheader.1 ]
  %inc11.1 = add nsw i64 %ca.37.1, 1
  br i1 false, label %for.body10.1, label %for.cond.loopexit.loopexit.1

for.cond.loopexit.loopexit.1:                     ; preds = %for.body10.1
  %inc11.lcssa.1 = phi i64 [ %inc11.1, %for.body10.1 ]
  br label %for.cond.loopexit.1

for.cond.loopexit.1:                              ; preds = %for.cond.loopexit.loopexit.1, %for.cond8.preheader.1
  %ca.3.lcssa.1 = phi i64 [ %ca.2.lcssa.1, %for.cond8.preheader.1 ], [ %inc11.lcssa.1, %for.cond.loopexit.loopexit.1 ]
  br i1 %cmp, label %for.cond5.preheader.1, label %for.end14.loopexit.1

for.end14.loopexit.1:                             ; preds = %for.cond.loopexit.1
  br label %for.end14.1

for.end14.1:                                      ; preds = %for.end14.loopexit.1, %for.end14
  ret void
}
