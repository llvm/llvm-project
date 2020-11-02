; If there's a call in the loop which dominates the backedge, we 
; don't need a safepoint poll (since the callee must contain a 
; poll test).
;; RUN: opt < %s -place-safepoints -S -enable-new-pm=0 | FileCheck %s

declare void @foo()

define void @test1() gc "statepoint-example" {
; CHECK-LABEL: test1

entry:
; CHECK-LABEL: entry
; CHECK: call void @do_safepoint
  br label %loop

loop:
; CHECK-LABEL: loop
; CHECK-NOT: call void @do_safepoint
  call void @foo()
  br label %loop
}

; This function is inlined when inserting a poll.
declare void @do_safepoint()
define void @gc.safepoint_poll() {
; CHECK-LABEL: gc.safepoint_poll
entry:
  call void @do_safepoint()
  ret void
}
