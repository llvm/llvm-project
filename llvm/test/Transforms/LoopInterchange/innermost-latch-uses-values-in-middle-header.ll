; REQUIRES: asserts
; RUN: opt < %s -basic-aa -loop-interchange -cache-line-size=64 -verify-dom-info -verify-loop-info \
; RUN:     -S -debug 2>&1 | FileCheck %s

@a = common global i32 0, align 4
@d = common dso_local local_unnamed_addr global [1 x [6 x i32]] zeroinitializer, align 4

;; After interchanging the innermost and the middle loop, we should not continue
;; doing interchange for the (new) middle loop and the outermost loop, because of
;; values defined in the new innermost loop not available in the exiting block of
;; the entire loop nest.
; CHECK: Loops are legal to interchange
; CHECK: Loops interchanged.
; CHECK: Found unsupported PHI nodes in inner loop latch.
; CHECK: Not interchanging loops. Cannot prove legality.
define void @innermost_latch_uses_values_in_middle_header() {
entry:
  %0 = load i32, i32* @a, align 4
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
  %arrayidx9.i = getelementptr inbounds [1 x [6 x i32]], [1 x [6 x i32]]* @d, i64 0, i64 %indvar.innermost, i64 %indvar.middle
  store i32 0, i32* %arrayidx9.i, align 4
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
