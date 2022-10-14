; Test the stacksave builtin.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare ptr@llvm.stacksave()

define void @f1(ptr %dest) {
; CHECK-LABEL: f1:
; CHECK: stg %r15, 0(%r2)
; CHECK: br %r14
  %addr = call ptr@llvm.stacksave()
  store volatile ptr %addr, ptr %dest
  ret void
}
