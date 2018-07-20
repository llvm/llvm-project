; RUN: llc -mtriple=x86_64-apple-macosx10.11 %s -o - | FileCheck %s

define void @test_unreachable() {
; CHECK-LABEL: test_unreachable:
; CHECK: ud2
  unreachable
}
