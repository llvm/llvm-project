; RUN: opt -passes='loop(indvars),print<scalar-evolution>' -S -disable-output < %s 2>&1 | FileCheck %s

; CHECK:  Determining loop execution counts for: @combined_bridge
; CHECK-NEXT:  Loop %inner2.ph: Unpredictable backedge-taken count.
; CHECK-NEXT:  Loop %inner2.ph: Unpredictable constant max backedge-taken count.
; CHECK-NEXT:  Loop %inner2.ph: Unpredictable symbolic max backedge-taken count.
; CHECK-NEXT:  Loop %inner1.ph: Unpredictable backedge-taken count.
; CHECK-NEXT:  Loop %inner1.ph: Unpredictable constant max backedge-taken count.
; CHECK-NEXT:  Loop %inner1.ph: Unpredictable symbolic max backedge-taken count.

define i64 @combined_bridge() {

entry:
  br label %inner1.ph

inner1.ph:                                      ; preds = %inner1.ph, %entry
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %inner1.ph ]
  %iv.next = add nsw i64 %iv, 1
  %exit.cond = icmp eq i64 %iv.next, 0
  br i1 %exit.cond, label %inner2.ph, label %inner1.ph

inner2.ph:                                      ; preds = %inner2.ph, %inner1.ph
  %iv2 = phi i64 [ %iv.next2, %inner2.ph ], [ 0, %inner1.ph ]
  %iv.next2 = add nsw i64 %iv2, 1
  %exit.cond2 = icmp eq i64 %iv.next2, 0
  br i1 %exit.cond2, label %outer2.exit.i, label %inner2.ph

outer2.exit.i:                                    ; preds = %inner2.ph
  ret i64 0
}

; CHECK:  Determining loop execution counts for: @indirect_cmp_user
; CHECK-NEXT:  Loop %loop: Unpredictable backedge-taken count.
; CHECK-NEXT:  Loop %loop: Unpredictable constant max backedge-taken count.
; CHECK-NEXT:  Loop %loop: Unpredictable symbolic max backedge-taken count.
define i64 @indirect_cmp_user() {

entry:
  br label %loop

loop:                                      ; preds = %latch, %entry
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %iv.next = add nsw i64 %iv, 1
  %cond = icmp eq i64 %iv.next, 0
  %exit.cond = or i1 %cond, false
  br label %latch

latch:
  br i1 %exit.cond, label %exit, label %loop
exit:                                    ; preds = %latch
  ret i64 0
}