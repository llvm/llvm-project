; Remove 'S' Scalar Dependencies #119345
; Scalar dependencies are not handled correctly, so they were removed to avoid
; miscompiles. The loop nest in this test case used to be interchanged, but it's
; no longer triggering. XFAIL'ing this test to indicate that this test should
; interchanged if scalar deps are handled correctly.
;
; XFAIL: *

; RUN: opt < %s -passes=loop-interchange -verify-dom-info -verify-loop-info -pass-remarks-output=%t -disable-output
; RUN: FileCheck -input-file %t %s

@a = common global i32 0, align 4
@d = common dso_local local_unnamed_addr global [1 x [6 x i32]] zeroinitializer, align 4

; After interchanging the innermost and the middle loop, we should not continue
; doing interchange for the (new) middle loop and the outermost loop, because of
; values defined in the new innermost loop not available in the exiting block of
; the entire loop nest.
;
; CHECK:  --- !Passed
; CHECK:  Pass:            loop-interchange
; CHECK:  Name:            Interchanged
; CHECK:  Function:        innermost_latch_uses_values_in_middle_header
; CHECK:  Args:
; CHECK:    - String:          Loop interchanged with enclosing loop.
; CHECK:  ...
; CHECK:  --- !Missed
; CHECK:  Pass:            loop-interchange
; CHECK:  Name:            UnsupportedInnerLatchPHI
; CHECK:  Function:        innermost_latch_uses_values_in_middle_header
; CHECK:  Args:
; CHECK:    - String:          Cannot interchange loops because unsupported PHI nodes found in inner loop latch.
; CHECK:  ...
; CHECK:  --- !Missed
; CHECK:  Pass:            loop-interchange
; CHECK:  Name:            UnsupportedExitPHI
; CHECK:  Function:        innermost_latch_uses_values_in_middle_header
; CHECK:  Args:
; CHECK:    - String:          Found unsupported PHI node in loop exit.
; CHECK:  ...
;
define void @innermost_latch_uses_values_in_middle_header() {
entry:
  %0 = load i32, ptr @a, align 4
  %b = add i32 80, 1
  br label %outermost.header

outermost.header:                      ; preds = %outermost.latch, %entry
  %indvar.outermost = phi i32 [ 10, %entry ], [ %indvar.outermost.next, %outermost.latch ]
  %tobool71.i = icmp eq i32 %0, 0
  br i1 %tobool71.i, label %middle.header, label %outermost.latch

middle.header:                            ; preds = %middle.latch, %outermost.header
  %indvar.middle = phi i64 [ 4, %outermost.header ], [ %indvar.middle.next, %middle.latch ]
  %indvar.middle.wide = zext i32 %b to i64 ; a def in the middle header
  br label %innermost.header

innermost.header:                                         ; preds = %middle.header, %innermost.latch
  %indvar.innermost = phi i64 [ %indvar.innermost.next, %innermost.latch ], [ 4, %middle.header ]
  br label %innermost.body

innermost.body:                                      ; preds = %innermost.header
  %arrayidx9.i = getelementptr inbounds [1 x [6 x i32]], ptr @d, i64 0, i64 %indvar.innermost, i64 %indvar.middle
  store i32 0, ptr %arrayidx9.i, align 4
  br label %innermost.latch

innermost.latch:                             ; preds = %innermost.body
  %indvar.innermost.next = add nsw i64 %indvar.innermost, 1
  %tobool5.i = icmp eq i64 %indvar.innermost.next, %indvar.middle.wide ; corresponding use in the innermost latch
  br i1 %tobool5.i, label %middle.latch, label %innermost.header

middle.latch:                                      ; preds = %innermost.latch
  %indvar.middle.next = add nsw i64 %indvar.middle, -1
  %tobool2.i = icmp eq i64 %indvar.middle.next, 0
  br i1 %tobool2.i, label %outermost.latch, label %middle.header

outermost.latch:                                      ; preds = %middle.latch, %outermost.header
  %indvar.outermost.next = add nsw i32 %indvar.outermost, -5
  %tobool.i = icmp eq i32 %indvar.outermost.next, 0
  br i1 %tobool.i, label %outermost.exit, label %outermost.header

outermost.exit:                                           ; preds = %outermost.latch
  ret void
}
