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
  %indvars.iv21 = phi i64 [ 0, %entry ], [ %indvars.iv.next22, %for.inc ]
  %arrayidx = getelementptr inbounds [8 x i32], ptr @arr2, i64 0, i64 %indvars.iv21
  %ld1 = load i32, ptr %arrayidx, align 4
  %0 = trunc i64 %indvars.iv21 to i32
  store i32 %0, ptr %arrayidx, align 4
  %1 = trunc i64 %indvars.iv21 to i32
  %add = add nsw i32 %1, %n
  %cmp.early = icmp eq i32 %ld1, 3
  br i1 %cmp.early, label %for.early, label %for.body.inner

for.body.inner:
  %indvars.iv = phi i64 [ 0, %for.body ], [ %indvars.iv.next, %for.body.inner ]
  %arrayidx7 = getelementptr inbounds [8 x [8 x i32]], ptr @arr, i64 0, i64 %indvars.iv, i64 %indvars.iv21
  store i32 %add, ptr %arrayidx7, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 8
  br i1 %exitcond, label %for.inc, label %for.body.inner

for.inc:
  %indvars.iv.next22 = add nuw nsw i64 %indvars.iv21, 1
  %exitcond23 = icmp eq i64 %indvars.iv.next22, 8
  br i1 %exitcond23, label %for.end, label %for.body, !llvm.loop !1

for.early:
  ret i32 1

for.end:
  ret i32 0
}

!1 = distinct !{!1, !2, !3}
!2 = !{!"llvm.loop.vectorize.width", i32 4}
!3 = !{!"llvm.loop.vectorize.enable", i1 true}
