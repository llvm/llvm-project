; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>" 2>&1 | FileCheck %s

; Verify that ScalarEvolution can compute a predicated backedge-taken count
; for a loop with a non-unit stride (stride = 3) and no 'nsw' flag on the
; induction variable (e.g. from compilation under -fwrapv).
; Without the predicate, the backedge-taken count is unpredictable.

define void @fwrapv_stride3(ptr noalias %x, i32 %l, i32 %u) {
; CHECK-LABEL: Determining loop execution counts for: @fwrapv_stride3
; CHECK-NEXT:  Loop %loop.body: Unpredictable backedge-taken count.
; CHECK-NEXT:  Loop %loop.body: Unpredictable constant max backedge-taken count.
; CHECK-NEXT:  Loop %loop.body: Unpredictable symbolic max backedge-taken count.
; CHECK-NEXT:  Loop %loop.body: Predicated backedge-taken count is
; CHECK-NEXT:   Predicates:
; CHECK-NEXT:      Compare predicate: %u sle) 2147483645

entry:
  %cmp1 = icmp slt i32 %l, %u
  br i1 %cmp1, label %loop.body, label %exit

loop.body:
  %i = phi i32 [ %l, %entry ], [ %i.next, %loop.body ]
  %idxprom = sext i32 %i to i64
  %arrayidx = getelementptr inbounds i32, ptr %x, i64 %idxprom
  %val = load i32, ptr %arrayidx, align 4
  %inc = add nsw i32 %val, 1
  store i32 %inc, ptr %arrayidx, align 4
  %i.next = add i32 %i, 3
  %cmp = icmp slt i32 %i.next, %u
  br i1 %cmp, label %loop.body, label %exit

exit:
  ret void
}
