; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize -disable-output < %s 2>&1 | FileCheck %s

; Check that the MaxInterleaveFactor for the target is reported in the debug output.

; CHECK: LV: MaxInterleaveFactor for the target is 1

define void @loop(ptr noalias %src, ptr noalias %dst, i64 %n) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %p = getelementptr i32, ptr %src, i64 %iv
  %v = load i32, ptr %p, align 4
  %q = getelementptr i32, ptr %dst, i64 %iv
  store i32 %v, ptr %q, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %cond = icmp ne i64 %iv.next, %n
  br i1 %cond, label %loop, label %exit

exit:
  ret void
}
