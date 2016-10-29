; RUN: opt -S -codegenprepare < %s | FileCheck %s

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Check that the smaller-width division that the BypassSlowDivision pass
; creates is not marked as "exact" (that is, it doesn't claim that the
; numerator is a multiple of the denominator).
;
; CHECK-LABEL: @test
define void @test(i64 %a, i64 %b, i64* %retptr) {
  ; CHECK: udiv i32
  %d = sdiv i64 %a, %b
  store i64 %d, i64* %retptr
  ret void
}
