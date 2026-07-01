; RUN: not opt < %s -S -passes=place-safepoints 2>&1 | FileCheck %s
; CHECK: LLVM ERROR: gc.safepoint_poll function is missing

define void @test_libcall() gc "statepoint-example" {
entry:
  ret void
}