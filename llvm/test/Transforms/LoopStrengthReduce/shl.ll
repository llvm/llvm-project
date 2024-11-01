; RUN: opt -opaque-pointers=0 < %s -loop-reduce -gvn -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"

define void @_Z3fooPfll(float* nocapture readonly %input, i64 %n, i64 %s) {
; CHECK-LABEL: @_Z3fooPfll(
entry:
  %mul = shl nsw i64 %s, 2
; CHECK: %mul = shl i64 %s, 2
  tail call void @_Z3bazl(i64 %mul) #2
; CHECK-NEXT: call void @_Z3bazl(i64 %mul)
  %cmp.5 = icmp sgt i64 %n, 0
  br i1 %cmp.5, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %i.06 = phi i64 [ %add, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds float, float* %input, i64 %i.06
; LoopStrengthReduce should reuse %mul as the stride.
; CHECK: getelementptr i1, i1* {{[^,]+}}, i64 %mul
  %0 = load float, float* %arrayidx, align 4
  tail call void @_Z3barf(float %0) #2
  %add = add nsw i64 %i.06, %s
  %cmp = icmp slt i64 %add, %n
  br i1 %cmp, label %for.body, label %for.cond.cleanup.loopexit
}

declare void @_Z3bazl(i64)

declare void @_Z3barf(float)
