; RUN: opt -passes='function(loop(indvars)),function(loop(loop-deletion)),function(loop(loop-reduce))' \
; RUN:   -S %s | FileCheck %s

; LoopDeletion partially hoists the computation of %conv7 while checking
; whether the inner loop is dead. Recompute SCEVs that use the newly invariant
; value so LSR can recognize %sext as an add recurrence.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @sideeffect(i32)

define void @test(i32 %val, i1 %c1) {
; CHECK-LABEL: define void @test(
; CHECK:       for.cond:
; CHECK:         [[START0:%.*]] = shl nuw nsw i32 %conv7, 24
; CHECK-NEXT:    [[START:%.*]] = add nuw nsw i32 [[START0]], 16777216
; CHECK:       for.body6:
; CHECK-NEXT:    [[SHIFT_IV:%.*]] = phi i32 [ [[SHIFT_IV_NEXT:%.*]], %for.body6 ], [ [[START]], %for.cond ]
; CHECK:         %conv8 = ashr exact i32 [[SHIFT_IV]], 24
; CHECK:         [[SHIFT_IV_NEXT]] = add i32 [[SHIFT_IV]], 16777216
; CHECK-NOT:     %h.039 =
; CHECK-NOT:     %inc =
; CHECK:         ret void
;
entry:
  br label %for.cond

for.cond:
  %f.0 = phi i32 [ 20, %entry ], [ 0, %for.inc11 ]
  br label %for.body6

for.body6:
  %h.039 = phi i32 [ 1, %for.cond ], [ %inc, %for.body6 ]
  %g.138 = phi i32 [ 0, %for.cond ], [ %and, %for.body6 ]
  %cmp = icmp eq i32 %val, -1
  %conv7 = zext i1 %cmp to i32
  %add.i = add nsw i32 %conv7, %h.039
  %sext = shl i32 %add.i, 24
  %conv8 = ashr exact i32 %sext, 24
  %cmp9 = icmp eq i32 %conv8, %f.0
  %conv10 = zext i1 %cmp9 to i32
  %and = add i32 %conv10, %g.138
  %inc = add i32 %h.039, 1
  %exitcond = icmp eq i32 %inc, 20000
  br i1 %exitcond, label %for.inc11, label %for.body6

for.inc11:
  %and.lcssa = phi i32 [ %and, %for.body6 ]
  call void @sideeffect(i32 %and.lcssa)
  br i1 %c1, label %for.cond, label %done

done:
  ret void
}
