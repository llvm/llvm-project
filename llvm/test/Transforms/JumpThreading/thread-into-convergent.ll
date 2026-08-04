; RUN: opt -S -passes=jump-threading < %s | FileCheck %s

; Jump threading must not create a new entry edge into a block that contains
; convergent operations: the new edge changes the control-flow paths by which
; threads reach the convergent call, and targets using the token-less
; `convergent` attribute  derive warp reconvergence points from
; the CFG merge structure. Threading past the %retest merge leaves no valid
; reconvergence point between the divergence introduced in %A/%C and the
; convergent call, so the call executes with a partially converged warp.

declare void @convergent_op() convergent
declare void @plain_op()

; %t and %f are opaque (runtime-uniform) conditions. On the %skip path %t is
; known false, so jump threading wants to duplicate %mid+%retest and route
; that path directly into %tail — a new entry edge into a convergent block.
; That must be refused. The %ret_body branch in between keeps the remaining
; structure from folding away.

define void @dont_thread_into_convergent(i1 %t, i1 %f) {
; CHECK-LABEL: @dont_thread_into_convergent(
; CHECK: retest:
; CHECK: br i1 %t, label %C, label %tail
; CHECK: tail:
; CHECK-NOT: tail.thread
; CHECK: call void @convergent_op()
entry:
  br i1 %t, label %A, label %skip

A:
  call void @plain_op()
  br label %mid

skip:
  br label %mid

mid:
  br i1 %f, label %ret_body, label %retest

ret_body:
  call void @plain_op()
  ret void

retest:
  br i1 %t, label %C, label %tail

C:
  call void @plain_op()
  br label %tail

tail:
  call void @convergent_op()
  ret void
}

; Same shape without the convergent call: threading fires and %tail2 gets the
; threaded entry edge. Guards against the new check being overly broad.

define void @thread_into_plain(i1 %t, i1 %f) {
; CHECK-LABEL: @thread_into_plain(
; CHECK: mid2.thread:
; CHECK: tail2:
; CHECK-SAME: preds = %mid2.thread
entry:
  br i1 %t, label %A2, label %skip2

A2:
  call void @plain_op()
  br label %mid2

skip2:
  br label %mid2

mid2:
  br i1 %f, label %ret_body2, label %retest2

ret_body2:
  call void @plain_op()
  ret void

retest2:
  br i1 %t, label %C2, label %tail2

C2:
  call void @plain_op()
  br label %tail2

tail2:
  call void @plain_op()
  ret void
}
