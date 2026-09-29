; RUN: llvm-as < %s >%t1
; RUN: llvm-lto -list-symbols-only %t1 | FileCheck %s

; This tests that we don't crash trying to find all uses in a lazily loaded
; module.
; CHECK: foo

target triple = "x86_64-unknown-linux-gnu"
define linkonce_odr void @foo() {
  ret void
}
