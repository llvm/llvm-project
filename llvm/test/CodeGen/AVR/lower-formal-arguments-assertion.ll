; RUN: llc < %s -mtriple=avr | FileCheck %s

define void @foo(i1) {
; CHECK-LABEL: foo:
; CHECK: ret
  ret void
}
