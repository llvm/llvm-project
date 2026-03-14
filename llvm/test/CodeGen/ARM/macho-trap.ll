; RUN: llc -mtriple=armv7-apple-ios7.0 %s -o - | FileCheck %s
; RUN: llc -mtriple=thumbv7-apple-ios7.0 %s -o - | FileCheck %s
; RUN: llc -mtriple=thumbv7m-apple-macho %s -o - | FileCheck %s
; RUN: llc -mtriple=thumbv6m-apple-macho %s -o - | FileCheck %s

define void @test_unreachable() {
; CHECK-LABEL: test_unreachable:
; CHECK: trap
  unreachable
}
