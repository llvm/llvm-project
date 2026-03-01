; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize -disable-output < %s 2>&1 | FileCheck %s

; Check that the MaxInterleaveFactor for the target is reported in the debug output.

; CHECK: LV: MaxInterleaveFactor for the target is 1

define void @loop(ptr noalias %src, ptr noalias %dst, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %p = getelementptr i32, ptr %src, i64 %i
  %v = load i32, ptr %p, align 4
  %q = getelementptr i32, ptr %dst, i64 %i
  store i32 %v, ptr %q, align 4
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp ne i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}
