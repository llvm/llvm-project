; RUN: opt -passes=loop-vectorize -mtriple=arm64-apple-macosx -mcpu=apple-m1 \
; RUN:     -pass-remarks=loop-vectorize -pass-remarks-analysis=loop-vectorize \
; RUN:     -disable-output %s 2>&1 | FileCheck %s

; The trip counts of the loops below depend on %stride.

; TODO: Currently VF and IC are chosen based on the original loop's trip count
; and the vector loop never executes.

; Trip count is 1 with %stride == 1, the vector loop can never execute.
; CHECK:      remark: {{.*}} vectorized loop (vectorization width: 2, interleaved count: 4)
; CHECK-NEXT: remark: {{.*}} Created vector loop never executes due to insufficient trip count.
define void @tc_1(ptr noalias %dst, ptr %src, i64 %stride) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.next = add nsw i64 %iv, 1
  %idx = mul i64 %iv, %stride
  %gep.src = getelementptr i64, ptr %src, i64 %idx
  %l = load i64, ptr %gep.src, align 8
  %gep.dst = getelementptr i64, ptr %dst, i64 %iv
  store i64 %l, ptr %gep.dst, align 8
  %ec = icmp slt i64 %iv.next, %stride
  br i1 %ec, label %loop, label %exit

exit:
  ret void
}

; Trip count is 4 with %stride == 1.
; CHECK:      remark: {{.*}} vectorized loop (vectorization width: 2, interleaved count: 4)
; CHECK-NEXT: remark: {{.*}} Created vector loop never executes due to insufficient trip count.
define void @tc_4(ptr noalias %dst, ptr %src, i64 %stride) {
entry:
  %n = shl i64 %stride, 2
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.next = add nsw i64 %iv, 1
  %idx = mul i64 %iv, %stride
  %gep.src = getelementptr i64, ptr %src, i64 %idx
  %l = load i64, ptr %gep.src, align 8
  %gep.dst = getelementptr i64, ptr %dst, i64 %iv
  store i64 %l, ptr %gep.dst, align 8
  %ec = icmp slt i64 %iv.next, %n
  br i1 %ec, label %loop, label %exit

exit:
  ret void
}

; The trip count is at most 12 via the loop guard, but VF * IC = 16.
; TODO: The main vector loop is created although it can never execute.
; CHECK: remark: <unknown>:0:0: vectorized loop (vectorization width: 4, interleaved count: 4)
define void @tc_bounded_by_ult_guard(ptr %dst, i64 %n) {
entry:
  %pre = icmp ult i64 %n, 13
  br i1 %pre, label %loop, label %exit

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep = getelementptr inbounds i32, ptr %dst, i64 %iv
  store i32 1, ptr %gep
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, %n
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}

; The trip count is at most 23 via the loop guard, but VF * IC = 32.
; TODO: The main vector loop is created although it can never execute.
; CHECK: remark: <unknown>:0:0: vectorized loop (vectorization width: 8, interleaved count: 4)
define void @tc_bounded_by_ult_guard_i16(ptr %dst, i64 %n) {
entry:
  %pre = icmp ult i64 %n, 24
  br i1 %pre, label %loop, label %exit

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep = getelementptr inbounds i16, ptr %dst, i64 %iv
  store i16 1, ptr %gep
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, %n
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}

; Loops whose trip count is bounded by a loop guard below the minimum
; profitable trip count.

; The trip count is at most 9, and the minimum profitable trip count is 10.
; CHECK: remark: <unknown>:0:0: vectorized loop (vectorization width: 2, interleaved count: 4)
; CHECK-NEXT: remark: <unknown>:0:0: Created vector loop never executes due to insufficient trip count.
define void @tc_below_min_profitable_tc(ptr %dst, ptr %a, ptr %b, ptr %c, i64 %n) {
entry:
  %g = icmp ult i64 %n, 10
  br i1 %g, label %loop, label %exit

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.a = getelementptr inbounds i64, ptr %a, i64 %iv
  %la = load i64, ptr %gep.a, align 8
  %gep.b = getelementptr inbounds i64, ptr %b, i64 %iv
  %lb = load i64, ptr %gep.b, align 8
  %gep.c = getelementptr inbounds i64, ptr %c, i64 %iv
  %lc = load i64, ptr %gep.c, align 8
  %add = add i64 %la, %lb
  %add2 = add i64 %add, %lc
  %gep.dst = getelementptr inbounds i64, ptr %dst, i64 %iv
  store i64 %add2, ptr %gep.dst, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %ec = icmp eq i64 %iv.next, %n
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}

; The trip count is at most 10, and the minimum profitable trip count is 10.
; CHECK: remark: <unknown>:0:0: vectorized loop (vectorization width: 2, interleaved count: 4)
define void @tc_may_be_min_profitable_tc(ptr %dst, ptr %a, ptr %b, ptr %c, i64 %n) {
entry:
  %g = icmp ult i64 %n, 11
  br i1 %g, label %loop, label %exit

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.a = getelementptr inbounds i64, ptr %a, i64 %iv
  %la = load i64, ptr %gep.a, align 8
  %gep.b = getelementptr inbounds i64, ptr %b, i64 %iv
  %lb = load i64, ptr %gep.b, align 8
  %gep.c = getelementptr inbounds i64, ptr %c, i64 %iv
  %lc = load i64, ptr %gep.c, align 8
  %add = add i64 %la, %lb
  %add2 = add i64 %add, %lc
  %gep.dst = getelementptr inbounds i64, ptr %dst, i64 %iv
  store i64 %add2, ptr %gep.dst, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %ec = icmp eq i64 %iv.next, %n
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}

; A scalar epilogue is required by the interleave group for the strided load of
; %a.
; CHECK: remark: <unknown>:0:0: vectorized loop (vectorization width: 2, interleaved count: 4)
; CHECK-NEXT: remark: <unknown>:0:0: Created vector loop never executes due to insufficient trip count.
define void @tc_equal_min_profitable_tc_requires_scalar_epilogue(ptr %dst, ptr %a, ptr %b, ptr %c, i64 %n) {
entry:
  %g = icmp ult i64 %n, 13
  br i1 %g, label %loop, label %exit

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %iv2 = shl nuw nsw i64 %iv, 1
  %gep.a = getelementptr inbounds i64, ptr %a, i64 %iv2
  %la = load i64, ptr %gep.a, align 8
  %gep.b = getelementptr inbounds i64, ptr %b, i64 %iv
  %lb = load i64, ptr %gep.b, align 8
  %gep.c = getelementptr inbounds i64, ptr %c, i64 %iv
  %lc = load i64, ptr %gep.c, align 8
  %add = add i64 %la, %lb
  %add2 = add i64 %add, %lc
  %gep.dst = getelementptr inbounds i64, ptr %dst, i64 %iv
  store i64 %add2, ptr %gep.dst, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %ec = icmp eq i64 %iv.next, %n
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}

; CHECK-NOT: Created vector loop never executes
