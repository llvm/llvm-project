; RUN: opt -S -place-safepoints < %s | FileCheck %s

define void @test() gc "statepoint-example" {
; CHECK-LABEL: test(
entry:
; CHECK: entry:
; CHECK: call void @do_safepoint()
  br label %other

other:
; CHECK: other:
  call void undef() "gc-leaf-function"
; CHECK: call void @do_safepoint()
  br label %other
}

declare void @do_safepoint()
define void @gc.safepoint_poll() {
  call void @do_safepoint()
  ret void
}
