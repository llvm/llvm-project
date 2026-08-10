; RUN: llc -mtriple=aarch64-linux-gnu -O0 -verify-machineinstrs -o - %s | FileCheck %s

define i1 @test_crash_i256(i256 %int) {
; CHECK-LABEL: test_crash_i256
; CHECK: ret
  ret i1 true
}
