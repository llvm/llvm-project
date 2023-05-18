; A simple, barebones test to check whether assembly can be emitted
; for the z/OS target
; RUN: llc < %s -mtriple=s390x-ibm-zos | FileCheck %s

@a = global i32 0, align 4

define signext i32 @main() {
; CHECK: .section ".text"
; CHECK: main:
; CHECK: .section "a"
entry:
  ret i32 0
}
