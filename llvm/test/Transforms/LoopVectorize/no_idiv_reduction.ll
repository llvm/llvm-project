; RUN: opt -passes=loop-vectorize -force-vector-width=2 -force-vector-interleave=1 -S < %s | FileCheck %s
@a = common global [128 x i32] zeroinitializer, align 16

;; Must not vectorize division reduction. Division is lossy.
define i32 @g() {
entry:
  br label %for.body

for.body:
  ; CHECK-LABEL: @g(
  ; CHECK-NOT: sdiv <2 x i32>
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %r.05 = phi i32 [ 80, %entry ], [ %div, %for.body ]
  %arrayidx = getelementptr inbounds [128 x i32], [128 x i32]* @a, i64 0, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %div = sdiv i32 %r.05, %0
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 %div
}
