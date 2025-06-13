; This test checks that we don't emit both
; successful and unsuccessful message about vectorization.

; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -debug -disable-output < %s 2>&1 | FileCheck %s
; CHECK-NOT: LV: We can vectorize this loop
; CHECK: LV: Not vectorizing: Cannot prove legality
; CHECK-NOT: LV: We can vectorize this loop

@a = global [32000 x i32] zeroinitializer, align 4

define void @foo(i32 %val1, i32 %val2) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %0 = phi i32 [ %val1, %entry ], [ %add1, %for.body ]
  %1 = phi i32 [ %val2, %entry ], [ %2, %for.body ]
  %iv = phi i64 [ 1, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds [32000 x i32], ptr @a, i64 0, i64 %iv
  %iv.next = add nuw nsw i64 %iv, 1
  %arrayidx2 = getelementptr inbounds [32000 x i32], ptr @a, i64 0, i64 %iv.next
  %2 = load i32, ptr %arrayidx2, align 4
  %add0 = add nsw i32 %2, %1
  %add1 = add nsw i32 %add0, %0
  store i32 %add1, ptr %arrayidx, align 4
  %exitcond = icmp eq i64 %iv.next, 31999
  br i1 %exitcond, label %exit, label %for.body

exit:                                 ; preds = %for.body
  ret void
}
