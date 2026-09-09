; REQUIRES: asserts
; RUN: not --crash opt < %s -S -passes=place-safepoints 2>&1 | FileCheck %s
; CHECK: gc.safepoint_poll must be a non-empty function

define void @test_libcall() gc "statepoint-example" {
entry:
  ret void
}

declare void @gc.safepoint_poll()