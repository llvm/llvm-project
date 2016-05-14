; RUN: opt < %s -S -loop-unroll -unroll-max-iteration-count-to-analyze=100 -unroll-dynamic-cost-savings-discount=1000 -unroll-threshold=10 -unroll-percent-dynamic-cost-saved-threshold=60 | FileCheck %s
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; When examining gep-instructions we shouldn't consider them simplified if the
; corresponding memory access isn't simplified. Doing the opposite might bias
; our estimate, so that we might decide to unroll even a simple memcpy loop.
;
; Thus, the following loop shouldn't be unrolled:
; CHECK-LABEL: @not_simplified_geps
; CHECK: br i1 %
; CHECK: ret void
define void @not_simplified_geps(i32* noalias %b, i32* noalias %c) {
entry:
  br label %for.body

for.body:
  %iv.0 = phi i64 [ 0, %entry ], [ %iv.1, %for.body ]
  %arrayidx1 = getelementptr inbounds i32, i32* %b, i64 %iv.0
  %x1 = load i32, i32* %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %c, i64 %iv.0
  store i32 %x1, i32* %arrayidx2, align 4
  %iv.1 = add nuw nsw i64 %iv.0, 1
  %exitcond = icmp eq i64 %iv.1, 10
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}
