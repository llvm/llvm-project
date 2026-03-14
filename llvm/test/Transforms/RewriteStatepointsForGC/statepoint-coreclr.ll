; RUN: opt < %s -S -passes=rewrite-statepoints-for-gc | FileCheck %s

; Basic test to make sure that safepoints are placed
; for CoreCLR GC

declare void @foo()

define void @test_simple_call() gc "coreclr" {
; CHECK-LABEL: test_simple_call
entry:
  br label %other
other:
; CHECK-LABEL: other
; CHECK: statepoint
; CHECK-NOT: gc.result
  call void @foo()
  ret void
}

; This function is inlined when inserting a poll.  To avoid recursive
; issues, make sure we don't place safepoints in it.
declare void @do_safepoint()
define void @gc.safepoint_poll() {
; CHECK-LABEL: gc.safepoint_poll
; CHECK-LABEL: entry
; CHECK-NEXT: do_safepoint
; CHECK-NEXT: ret void
entry:
  call void @do_safepoint()
  ret void
}
