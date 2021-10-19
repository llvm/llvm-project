; RUN: opt < %s -S -place-safepoints -enable-new-pm=0 | FileCheck %s


; Do we insert a simple entry safepoint?
define void @test_entry() gc "statepoint-example" {
; CHECK-LABEL: @test_entry
entry:
; CHECK-LABEL: entry
; CHECK: call void @do_safepoint
  ret void
}

; On a non-gc function, we should NOT get an entry safepoint
define void @test_negative() {
; CHECK-LABEL: @test_negative
entry:
; CHECK-NOT: do_safepoint
  ret void
}

; Do we insert a backedge safepoint in a statically
; infinite loop?
define void @test_backedge() gc "statepoint-example" {
; CHECK-LABEL: test_backedge
entry:
; CHECK-LABEL: entry
; This statepoint is technically not required, but we don't exploit that yet.
; CHECK: call void @do_safepoint
  br label %other

; CHECK-LABEL: other
; CHECK: call void @do_safepoint
other:
  br label %other
}

; Check that we remove an unreachable block rather than trying
; to insert a backedge safepoint
define void @test_unreachable() gc "statepoint-example" {
; CHECK-LABEL: test_unreachable
entry:
; CHECK-LABEL: entry
; CHECK: call void @do_safepoint
  ret void

; CHECK-NOT: other
; CHECK-NOT: do_safepoint
other:
  br label %other
}

declare void @foo()

declare zeroext i1 @i1_return_i1(i1)

define i1 @test_call_with_result() gc "statepoint-example" {
; CHECK-LABEL: test_call_with_result
; This is checking that a statepoint_poll is inserted for a function
; that takes 1 argument.
; CHECK: call void @do_safepoint
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
