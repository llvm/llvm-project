; RUN: opt -S -mtriple=s390x-unknown-linux -mcpu=z16 -passes=loop-vectorize < %s \
; RUN:   | FileCheck %s
;
; Test that loop vectorizer generates a vectorized epilogue loop after a VF16
; vectorization.

define void @fun(ptr %Src, ptr %Dst, i64 %wide.trip.count) {
; CHECK-LABEL: @fun(
; CHECK-LABEL: vector.body:
; CHECK: %wide.load = load <16 x i8>
; CHECK-LABEL: vec.epilog.vector.body:
; CHECK: %wide.load8 = load <4 x i8>
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %indvars.iv.next = add i64 %indvars.iv, 1
  %arrayidx0 = getelementptr i8, ptr %Src, i64 %indvars.iv
  %0 = load i8, ptr %arrayidx0
  %arrayidx1 = getelementptr i8, ptr %Dst, i64 %indvars.iv
  store i8 %0, ptr %arrayidx1
  %exitcond.not = icmp eq i64 %indvars.iv, %wide.trip.count
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret void
}
