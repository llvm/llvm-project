; RUN: opt -passes=loop-unroll -debug-only=loop-unroll -disable-output < %s
; REQUIRES: asserts
; Compile this test with debug flag on to verify domtree right after loop unrolling.
target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"

; PR27157
define void @foo(i1 %arg) {
entry:
  br label %loop_header
loop_header:
  %iv = phi i64 [ 0, %entry ], [ %iv_next, %loop_latch ]
  br i1 %arg, label %loop_latch, label %loop_exiting_bb1
loop_exiting_bb1:
  br i1 false, label %loop_exiting_bb2, label %exit1.loopexit
loop_exiting_bb2:
  br i1 false, label %loop_latch, label %bb
bb:
  br label %exit1
loop_latch:
  %iv_next = add nuw nsw i64 %iv, 1
  %cmp = icmp ne i64 %iv_next, 2
  br i1 %cmp, label %loop_header, label %exit2
exit1.loopexit:
  br label %exit1
exit1:
  ret void
exit2:
  ret void
}

define void @foo2(i1 %arg) {
entry:
  br label %loop.header
loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.inc, %latch ]
  %iv.inc = add i32 %iv, 1
  br i1 %arg, label %diamond, label %latch
diamond:
  br i1 %arg, label %left, label %right
left:
  br i1 %arg, label %exit, label %merge
right:
  br i1 %arg, label %exit, label %merge
merge:
  br label %latch
latch:
  %end.cond = icmp eq i32 %iv, 1
  br i1 %end.cond, label %exit1, label %loop.header
exit:
  ret void
exit1:
  ret void
}
