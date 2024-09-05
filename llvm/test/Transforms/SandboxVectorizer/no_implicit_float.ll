; RUN: opt -passes=sandbox-vectorizer -debug %s 2>&1 | FileCheck %s
; REQUIRES: asserts
; Please note that this won't update automatically with update_test_checks.py !

; Check that we early return if the function has the NoImplicitFloat attribute.
define void @no_implicit_float() noimplicitfloat {
; CHECK: SBVec: NoImplicitFloat attribute, return.
  ret void
}
