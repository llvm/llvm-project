; RUN: llc -mtriple=x86_64-none-linux -fast-isel -fast-isel-abort=1 -verify-machineinstrs < %s | FileCheck %s

define i64 @test_sdiv64(i64 %dividend, i64 %divisor) nounwind {
entry:
  %result = sdiv i64 %dividend, %divisor
  ret i64 %result
}

; CHECK-LABEL: test_sdiv64:
; CHECK: cqto
; CHECK: idivq

define i64 @test_srem64(i64 %dividend, i64 %divisor) nounwind {
entry:
  %result = srem i64 %dividend, %divisor
  ret i64 %result
}

; CHECK-LABEL: test_srem64:
; CHECK: cqto
; CHECK: idivq

define i64 @test_udiv64(i64 %dividend, i64 %divisor) nounwind {
entry:
  %result = udiv i64 %dividend, %divisor
  ret i64 %result
}

; CHECK-LABEL: test_udiv64:
; CHECK: xorl
; CHECK: divq

define i64 @test_urem64(i64 %dividend, i64 %divisor) nounwind {
entry:
  %result = urem i64 %dividend, %divisor
  ret i64 %result
}

; CHECK-LABEL: test_urem64:
; CHECK: xorl
; CHECK: divq
