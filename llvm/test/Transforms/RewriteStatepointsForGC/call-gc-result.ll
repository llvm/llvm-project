;; RUN: opt < %s -rewrite-statepoints-for-gc -S | FileCheck %s

;; This test is to verify that gc_result from a call statepoint
;; can have preceding phis in its parent basic block. Unlike
;; invoke statepoint, call statepoint does not terminate the
;; block, and thus its gc_result is in the same block with the
;; call statepoint.

declare i32 @foo()

define i32 @test1(i1 %cond, i32 %a) gc "statepoint-example" {
entry:
  br i1 %cond, label %branch1, label %branch2
  
branch1:
  %b = add i32 %a, 1
  br label %merge
 
branch2:
  br label %merge

merge:
;; CHECK: 		%phi = phi i32 [ %a, %branch2 ], [ %b, %branch1 ]
;; CHECK-NEXT:  [[TOKEN:%[^ ]+]] = call token (i64, i32, i32 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i32f(i64 2882400000, i32 0, i32 ()* @foo, i32 0, i32 0, i32 0, i32 0
;; CHECK-NEXT:  call i32 @llvm.experimental.gc.result.i32(token [[TOKEN]])
  %phi = phi i32 [ %a, %branch2 ], [ %b, %branch1 ]
  %ret = call i32 @foo()
  ret i32 %ret
}

; This function is inlined when inserting a poll.
declare void @do_safepoint()
define void @gc.safepoint_poll() {
; CHECK-LABEL: gc.safepoint_poll
entry:
  call void @do_safepoint()
  ret void
}
