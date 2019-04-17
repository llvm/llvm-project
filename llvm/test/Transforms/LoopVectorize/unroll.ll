; This test makes sure that loop will not be unrolled in vectorization if VF computed
; equals to 1.
; RUN: opt < %s -loop-vectorize -S | FileCheck %s

; Make sure there are no geps being merged.
; CHECK-LABEL: @foo(
; CHECK: getelementptr
; CHECK-NOT: getelementptr

@N = common global i32 0, align 4
@a = common global [1000 x i32] zeroinitializer, align 16

define void @foo() #0 {
entry:
  %0 = load i32, i32* @N, align 4
  %cmp5 = icmp sgt i32 %0, 0
  br i1 %cmp5, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  %conv = sext i32 %0 to i64
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.06 = phi i64 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %mul = mul nuw nsw i64 %i.06, 7
  %arrayidx = getelementptr inbounds [1000 x i32], [1000 x i32]* @a, i64 0, i64 %mul
  store i32 3, i32* %arrayidx, align 4
  %inc = add nuw nsw i64 %i.06, 1
  %cmp = icmp slt i64 %inc, %conv
  br i1 %cmp, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}
