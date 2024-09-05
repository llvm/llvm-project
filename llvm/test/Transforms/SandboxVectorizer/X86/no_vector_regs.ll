; RUN: opt -passes=sandbox-vectorizer -debug -mtriple=x86_64-- -mattr=-sse %s 2>&1 | FileCheck %s
; REQUIRES: asserts
; Please note that this won't update automatically with update_test_checks.py !

; Check that we early return if the target has no vector registers.
define void @no_vector_regs() {
; CHECK: SBVec: Target has no vector registers, return.
  ret void
}
