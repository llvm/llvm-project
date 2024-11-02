; RUN: opt < %s -passes=loop-vectorize -force-vector-width=4 -verify-scev-maps -S |FileCheck %s

; SCEV expansion uses existing value when the SCEV has no AddRec expr.
; CHECK-LABEL: @foo(
; CHECK: select
; CHECK-NOT: select
; CHECK: ret

@a = common global [1000 x i16] zeroinitializer, align 16

define i32 @foo(i32 %x, i32 %y) {
entry:
  %cmp = icmp slt i32 %x, %y
  %cond = select i1 %cmp, i32 %x, i32 %y
  %cmp1.10 = icmp sgt i32 %cond, 0
  br i1 %cmp1.10, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  %tmp = sext i32 %cond to i64
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %indvars.iv = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next, %for.body ]
  %total.011 = phi i32 [ 0, %for.body.lr.ph ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds [1000 x i16], [1000 x i16]* @a, i64 0, i64 %indvars.iv
  %tmp1 = load i16, i16* %arrayidx, align 2
  %conv = sext i16 %tmp1 to i32
  %add = add nsw i32 %conv, %total.011
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %cmp1 = icmp slt i64 %indvars.iv.next, %tmp
  br i1 %cmp1, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body
  %add.lcssa = phi i32 [ %add, %for.body ]
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  %total.0.lcssa = phi i32 [ 0, %entry ], [ %add.lcssa, %for.end.loopexit ]
  ret i32 %total.0.lcssa
}
