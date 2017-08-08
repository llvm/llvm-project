; RUN: opt %s -loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -S | FileCheck %s
; RUN: opt %s -loop-vectorize -force-vector-interleave=2 -force-vector-width=4 -S | FileCheck %s -check-prefix=UNROLL

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@b = common global [1000 x i32] zeroinitializer, align 16
@c = common global [1000 x i32] zeroinitializer, align 16
@a = common global [1000 x i32] zeroinitializer, align 16

; Generate min.iters.check to skip the vector loop and jump to scalar.ph directly when loop iteration number is less than VF * UF.
; CHECK-LABEL: foo(
; CHECK: %min.iters.check = icmp ult i64 %N, 4
; CHECK: br i1 %min.iters.check, label %scalar.ph, label %vector.ph
; UNROLL-LABEL: foo(
; UNROLL: %min.iters.check = icmp ult i64 %N, 8
; UNROLL: br i1 %min.iters.check, label %scalar.ph, label %vector.ph

define void @foo(i64 %N) {
entry:
  %cmp.8 = icmp sgt i64 %N, 0
  br i1 %cmp.8, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.preheader
  %i.09 = phi i64 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds [1000 x i32], [1000 x i32]* @b, i64 0, i64 %i.09
  %tmp = load i32, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds [1000 x i32], [1000 x i32]* @c, i64 0, i64 %i.09
  %tmp1 = load i32, i32* %arrayidx1, align 4
  %add = add nsw i32 %tmp1, %tmp
  %arrayidx2 = getelementptr inbounds [1000 x i32], [1000 x i32]* @a, i64 0, i64 %i.09
  store i32 %add, i32* %arrayidx2, align 4
  %inc = add nuw nsw i64 %i.09, 1
  %exitcond = icmp eq i64 %inc, %N
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}
