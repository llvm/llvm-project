; REQUIRES: asserts
; RUN: not --crash opt < %s -S -passes=place-safepoints 2>&1 | FileCheck %s
; CHECK: gc.safepoint_poll declared with wrong type

define void @test_libcall() gc "statepoint-example" {
entry:
  ret void
}

define i32 @gc.safepoint_poll() {
entry:
  ret i32 0
}