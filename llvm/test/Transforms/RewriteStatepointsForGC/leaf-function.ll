; RUN: opt < %s -S -rewrite-statepoints-for-gc | FileCheck %s

declare void @foo() "gc-leaf-function"
declare void @bar()

; Calls of functions with the "gc-leaf-function" attribute shouldn't be turned
; into a safepoint.  An entry safepoint should get inserted, though.
define void @test_leaf_function() gc "statepoint-example" {
; CHECK-LABEL: test_leaf_function
; CHECK-NOT: gc.statepoint
; CHECK-NOT: gc.result
entry:
  call void @foo()
  ret void
}

define void @test_leaf_function_call() gc "statepoint-example" {
; CHECK-LABEL: test_leaf_function_call
; CHECK-NOT: gc.statepoint
; CHECK-NOT: gc.result
entry:
  call void @bar() "gc-leaf-function"
  ret void
}

; This function is inlined when inserting a poll.
declare void @do_safepoint()
define void @gc.safepoint_poll() {
; CHECK-LABEL: gc.safepoint_poll
entry:
  call void @do_safepoint()
  ret void
}
