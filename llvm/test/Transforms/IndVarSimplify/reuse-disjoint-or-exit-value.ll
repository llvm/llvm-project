; RUN: opt -passes='loop(indvars)' -S < %s | FileCheck %s

; Reuse the exit value by replacing its disjoint or with an add. Without reuse,
; SCEVExpander emits a multiply chain:
;
;   %0 = mul i24 %hi, 257
;   %1 = add i24 %0, 513
;
; Reduced from Transforms/PhaseOrdering/reuse-disjoint-or-exit-value.ll at the
; inner loop's indvars invocation. Keep the outer recurrence: without it,
; re-expansion only needs a trunc and an add and is cheaper than reuse.
;
; Check that indvars reuses %hi.trunc without a multiply. The phase-ordering
; test covers the subsequent folding and sinking of the reused chain.

define i1 @reuse_disjoint_or_exit_value(i32 %itr) {
; CHECK-LABEL: define i1 @reuse_disjoint_or_exit_value(
; CHECK-NOT:     mul
;
; Replace the or with an add and reuse %hi.trunc for the outer recurrence.
; CHECK:       [[OUTER_LOOPEXIT:.*]]:
; CHECK:         %or = add i24 %hi, 1
; CHECK:         %hi.trunc = trunc i32 %combined to i24
; CHECK-NOT:     mul
;
entry:
  %outer.cmp = icmp eq i32 %itr, 0
  br label %split

split:                                            ; preds = %entry, %outer.loopexit
  %hi = phi i24 [ 0, %entry ], [ %hi.next.lcssa, %outer.loopexit ]
  %or = or disjoint i24 %hi, 1
  %or.ext = zext i24 %or to i32
  %step = add nuw nsw i32 %or.ext, 1
  %step.hi = shl i32 %step, 8
  %lo.trunc = trunc i32 %step to i8
  %combined = add i32 %step.hi, %or.ext
  %hi.trunc = trunc i32 %combined to i24
  br label %inner.latch

inner.latch:                                      ; preds = %inner.latch, %split
  %hi.next = phi i24 [ %hi.trunc, %inner.latch ], [ 0, %split ]
  %lo.next = phi i8 [ %lo.trunc, %inner.latch ], [ 0, %split ]
  %inner.cmp = phi i1 [ false, %inner.latch ], [ true, %split ]
  br i1 %inner.cmp, label %inner.latch, label %outer.loopexit

outer.loopexit:                                   ; preds = %inner.latch
  %hi.next.lcssa = phi i24 [ %hi.next, %inner.latch ]
  %lo.next.lcssa = phi i8 [ %lo.next, %inner.latch ]
  br i1 %outer.cmp, label %split, label %exit

exit:                                             ; preds = %outer.loopexit
  %lo.lcssa = phi i8 [ %lo.next.lcssa, %outer.loopexit ]
  %res = icmp eq i8 %lo.lcssa, 0
  ret i1 %res
}
