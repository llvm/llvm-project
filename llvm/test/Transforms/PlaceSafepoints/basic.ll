; RUN: opt %s -S -place-safepoints | FileCheck %s


; Do we insert a simple entry safepoint?
define void @test_entry() gc "statepoint-example" {
; CHECK-LABEL: @test_entry
entry:
; CHECK-LABEL: entry
; CHECK: statepoint
  ret void
}

; On a non-gc function, we should NOT get an entry safepoint
define void @test_negative() {
; CHECK-LABEL: @test_negative
entry:
; CHECK-NOT: statepoint
  ret void
}

; Do we insert a backedge safepoint in a statically
; infinite loop?
define void @test_backedge() gc "statepoint-example" {
; CHECK-LABEL: test_backedge
entry:
; CHECK-LABEL: entry
; This statepoint is technically not required, but we don't exploit that yet.
; CHECK: statepoint
  br label %other

; CHECK-LABEL: other
; CHECK: statepoint
other:
  call void undef()
  br label %other
}

; Check that we remove an unreachable block rather than trying
; to insert a backedge safepoint
define void @test_unreachable() gc "statepoint-example" {
; CHECK-LABEL: test_unreachable
entry:
; CHECK-LABEL: entry
; CHECK: statepoint
  ret void

; CHECK-NOT: other
; CHECK-NOT: statepoint
other:
  br label %other
}

declare void @foo()

; Do we turn a call into it's own statepoint
define void @test_simple_call() gc "statepoint-example" {
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

declare zeroext i1 @i1_return_i1(i1)

define i1 @test_call_with_result() gc "statepoint-example" {
; CHECK-LABEL: test_call_with_result
; This is checking that a statepoint_poll + statepoint + result is
; inserted for a function that takes 1 argument.
; CHECK: gc.statepoint.p0f_isVoidf
; CHECK: gc.statepoint.p0f_i1i1f
; CHECK: (i64 2882400000, i32 0, i1 (i1)* @i1_return_i1, i32 1, i32 0, i1 false, i32 0, i32 0)
; CHECK: %call1.2 = call i1 @llvm.experimental.gc.result.i1
entry:
  %call1 = tail call i1 (i1) @i1_return_i1(i1 false)
  ret i1 %call1
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
