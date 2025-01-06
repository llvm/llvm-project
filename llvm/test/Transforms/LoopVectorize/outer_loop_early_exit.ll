; REQUIRES: asserts
; RUN: opt -S -passes=loop-vectorize -enable-vplan-native-path -disable-output -debug 2>&1 < %s | FileCheck %s

; CHECK-LABEL: LV: Found a loop: for.body
; CHECK: LV: Not vectorizing: Unsupported conditional branch.
; CHECK: loop not vectorized: loop control flow is not understood by vectorizer
; CHECK: LV: Not vectorizing: Unsupported outer loop.

@arr2 = external global [8 x i32], align 16
@arr = external global [8 x [8 x i32]], align 16

define i32 @foo(i32 %n) {
entry:
  br label %for.body

for.body:
  %iv.outer = phi i64 [ 0, %entry ], [%iv.outer.next, %for.inc ]
  %arrayidx = getelementptr inbounds [8 x i32], ptr @arr2, i64 0, i64 %iv.outer
  %ld1 = load i32, ptr %arrayidx, align 4
  %0 = trunc i64 %iv.outer to i32
  store i32 %0, ptr %arrayidx, align 4
  %1 = trunc i64 %iv.outer to i32
  %add = add nsw i32 %1, %n
  %cmp.early = icmp eq i32 %ld1, 3
  br i1 %cmp.early, label %for.early, label %for.body.inner

for.body.inner:
  %iv.inner = phi i64 [ 0, %for.body ], [ %iv.inner.next, %for.body.inner ]
  %arrayidx7 = getelementptr inbounds [8 x [8 x i32]], ptr @arr, i64 0, i64 %iv.inner, i64 %iv.outer
  store i32 %add, ptr %arrayidx7, align 4
  %iv.inner.next = add nuw nsw i64 %iv.inner, 1
  %cmp.inner = icmp eq i64 %iv.inner.next, 8
  br i1 %cmp.inner, label %for.inc, label %for.body.inner

for.inc:
  %iv.outer.next = add nuw nsw i64 %iv.outer, 1
  %cmp.outer = icmp eq i64%iv.outer.next, 8
  br i1 %cmp.outer, label %for.end, label %for.body, !llvm.loop !1

for.early:
  ret i32 1

for.end:
  ret i32 0
}

!1 = distinct !{!1, !2, !3}
!2 = !{!"llvm.loop.vectorize.width", i32 4}
!3 = !{!"llvm.loop.vectorize.enable", i1 true}
