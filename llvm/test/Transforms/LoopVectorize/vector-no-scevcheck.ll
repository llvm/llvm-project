; RUN: opt -passes=loop-vectorize -force-vector-width=2 -force-vector-interleave=1 -S < %s | FileCheck %s

; This test is to ensure that SCEV checks (which are costly performancewise) are
; not generated when appropriate aliasing checks are sufficient.

define void @foo(ptr %pout, ptr %pin, i64 %val0, i64 %val1, i64 %val2) {
; CHECK-LABEL: @foo(
; CHECK-NOT: vector.scevcheck
; CHECK: vector.body
entry:
  %0 = getelementptr double, ptr %pin, i64 %val0
  br label %loop1.header

loop1.header:                                     ; preds = %loop1.latch, %entry
  %i = phi i64 [ %i.next, %loop1.latch ], [ 0, %entry ]
  %mul0 = mul nsw i64 %i, %val2
  %arrayidx0 = getelementptr inbounds double, ptr %0, i64 %mul0
  %mul1 = mul nsw i64 %i, %val1
  br label %loop2.header

loop2.header:                                     ; preds = %loop1.header, %loop2.header
  %j = phi i64 [ 0, %loop1.header ], [ %j.next, %loop2.header ]
  %1 = load double, ptr %arrayidx0, align 8
  %arrayidx1 = getelementptr inbounds double, ptr %0, i64 %j
  %2 = load double, ptr %arrayidx1, align 8
  %sum = fadd contract double %1, %2
  %3 = getelementptr double, ptr %pout, i64 %mul1
  %arrayidx2 = getelementptr inbounds double, ptr %3, i64 %j
  store double %sum, ptr %arrayidx2, align 8
  %j.next = add nuw nsw i64 %j, 1
  %cmp = icmp slt i64 %j.next, %val1
  br i1 %cmp, label %loop2.header, label %loop1.latch

loop1.latch:                                      ; preds = %loop2.header
  %i.next = add nuw nsw i64 %i, 1
  %exitcond = icmp eq i64 %i.next, %val1
  br i1 %exitcond, label %exit, label %loop1.header

exit:                                             ; preds = %loop1.latch
  ret void
}

; Similar test to the above but with the %arrayidx0 moved to the loop2.header

define void @bar(ptr %pout, ptr %pin, i64 %val0, i64 %val1, i64 %val2) {
; CHECK-LABEL: @bar(
; CHECK-NOT: vector.scevcheck
; CHECK: vector.body
entry:
  %0 = getelementptr double, ptr %pin, i64 %val0
  br label %loop1.header

loop1.header:                                     ; preds = %loop1.latch, %entry
  %i = phi i64 [ %i.next, %loop1.latch ], [ 0, %entry ]
  %mul0 = mul nsw i64 %i, %val2
  %mul1 = mul nsw i64 %i, %val1
  br label %loop2.header

loop2.header:                                     ; preds = %loop1.header, %loop2.header
  %j = phi i64 [ 0, %loop1.header ], [ %j.next, %loop2.header ]
  %arrayidx0 = getelementptr inbounds double, ptr %0, i64 %mul0
  %1 = load double, ptr %arrayidx0, align 8
  %arrayidx1 = getelementptr inbounds double, ptr %0, i64 %j
  %2 = load double, ptr %arrayidx1, align 8
  %sum = fadd contract double %1, %2
  %3 = getelementptr double, ptr %pout, i64 %mul1
  %arrayidx2 = getelementptr inbounds double, ptr %3, i64 %j
  store double %sum, ptr %arrayidx2, align 8
  %j.next = add nuw nsw i64 %j, 1
  %cmp = icmp slt i64 %j.next, %val1
  br i1 %cmp, label %loop2.header, label %loop1.latch

loop1.latch:                                      ; preds = %loop2.header
  %i.next = add nuw nsw i64 %i, 1
  %exitcond = icmp eq i64 %i.next, %val1
  br i1 %exitcond, label %exit, label %loop1.header

exit:                                             ; preds = %loop1.latch
  ret void
}
