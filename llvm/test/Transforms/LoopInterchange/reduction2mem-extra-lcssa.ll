; Non-reduction LCSSA PHIs in the inner-loop exit must still be validated
; even when a reduction LCSSA PHI is present. Used to crash.
;
; RUN: opt < %s -passes=loop-interchange -loop-interchange-reduction-to-mem \
; RUN:   -pass-remarks-missed=loop-interchange -pass-remarks-output=%t -S \
; RUN:   | FileCheck -check-prefix=IR %s
; RUN: FileCheck --input-file=%t %s

; CHECK: --- !Missed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            UnsupportedExitPHI
; CHECK-NEXT: Function:        reduction_lcssa_with_non_phi_user
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          Found unsupported PHI node in loop exit.

@A = common global [100 x [100 x i32]] zeroinitializer
@sum = common global [100 x i32] zeroinitializer

; IR-LABEL: @reduction_lcssa_with_non_phi_user(
; IR-NOT: split
define void @reduction_lcssa_with_non_phi_user() {
entry:
  br label %outer.header

outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %sum.ptr = getelementptr inbounds [100 x i32], ptr @sum, i64 0, i64 %i
  br label %inner

inner:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner ]
  %red = phi i32 [ 0, %outer.header ], [ %add, %inner ]
  %ptr = getelementptr inbounds [100 x [100 x i32]], ptr @A, i64 0, i64 %j, i64 %i
  %val = load i32, ptr %ptr
  %add = add i32 %red, %val
  %j.next = add nuw nsw i64 %j, 1
  %exitcond.inner = icmp eq i64 %j, 99
  br i1 %exitcond.inner, label %inner.exit, label %inner

inner.exit:
  %red.lcssa = phi i32 [ %add, %inner ]
  %j.lcssa = phi i64 [ %j, %inner ]
  store i32 %red.lcssa, ptr %sum.ptr
  %use.j = add i64 %j.lcssa, 1
  br label %outer.latch

outer.latch:
  %i.next = add nuw nsw i64 %i, 1
  %exitcond.outer = icmp eq i64 %i, 99
  br i1 %exitcond.outer, label %exit, label %outer.header

exit:
  ret void
}
