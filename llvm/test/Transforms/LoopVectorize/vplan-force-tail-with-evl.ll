; REQUIRES: asserts

; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize \
; RUN: -force-tail-folding-style=data-with-evl -force-vector-width=4 \
; RUN: -force-target-supports-scalable-vectors -scalable-vectorization=on \
; RUN: -disable-output < %s 2>&1 | FileCheck --check-prefixes=NO-VP %s

; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize \
; RUN: -force-tail-folding-style=none \
; RUN: -prefer-predicate-over-epilogue=predicate-dont-vectorize \
; RUN: -force-target-supports-scalable-vectors -scalable-vectorization=on \
; RUN: -disable-output < %s 2>&1 | FileCheck --check-prefixes=NO-VP %s

; The target does not support predicated vectorization.
define void @foo(ptr noalias %a, ptr noalias %b, ptr noalias %c, i64 %N) {
; NO-VP-NOT: EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %b, i64 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %c, i64 %iv
  %1 = load i32, ptr %arrayidx2, align 4
  %add = add nsw i32 %1, %0
  %arrayidx4 = getelementptr inbounds i32, ptr %a, i64 %iv
  store i32 %add, ptr %arrayidx4, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret void
}

