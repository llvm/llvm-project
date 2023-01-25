; RUN: opt < %s -passes=loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -S |FileCheck %s
; SCEV expansion uses existing value or value + offset to reduce duplicate code expansion so foo should only generate one select inst after loop vectorization.
; CHECK-LABEL: @foo(
; CHECK: select
; CHECK-NOT: select

@ySrcL = common global ptr null, align 8
@smL = common global i32 0, align 4

define void @foo(i32 %rwL, i32 %kL, i32 %xfL) {
entry:
  %sub = add nsw i32 %rwL, -1
  %shr = ashr i32 %xfL, 6
  %cmp.i = icmp slt i32 %sub, %shr
  %cond.i = select i1 %cmp.i, i32 %sub, i32 %shr
  %cmp6 = icmp sgt i32 %cond.i, %kL
  br i1 %cmp6, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  %tmp = load ptr, ptr @ySrcL, align 8
  %tmp1 = sext i32 %kL to i64
  %tmp2 = sext i32 %cond.i to i64
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %indvars.iv = phi i64 [ %tmp1, %for.body.lr.ph ], [ %indvars.iv.next, %for.body ]
  %reduct.07 = phi i32 [ 0, %for.body.lr.ph ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %tmp, i64 %indvars.iv
  %tmp3 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %tmp3 to i32
  %add = add nsw i32 %conv, %reduct.07
  %indvars.iv.next = add nsw i64 %indvars.iv, 1
  %cmp = icmp slt i64 %indvars.iv.next, %tmp2
  br i1 %cmp, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body
  %add.lcssa = phi i32 [ %add, %for.body ]
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  %reduct.0.lcssa = phi i32 [ 0, %entry ], [ %add.lcssa, %for.end.loopexit ]
  store i32 %reduct.0.lcssa, ptr @smL, align 4
  ret void
}
