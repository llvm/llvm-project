; RUN: llc -mtriple=aarch64-apple-ios7.0 %s -o - | FileCheck %s

define void @test_unreachable() {
; CHECK-LABEL: test_unreachable:
; CHECK: brk #0x1
  unreachable
}
