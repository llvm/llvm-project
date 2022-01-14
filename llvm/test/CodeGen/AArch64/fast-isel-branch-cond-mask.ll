; RUN: llc -mtriple=aarch64-apple-darwin -O0 -fast-isel -fast-isel-abort=0 -verify-machineinstrs < %s | FileCheck %s

define void @test(i64 %a, i64 %b, i2* %c) {
; CHECK-LABEL: test
; CHECK:       and [[REG1:w[0-9]+]], {{w[0-9]+}}, #0x3
; CHECK-NEXT:  strb [[REG1]], [x2]
; CHECK-NEXT:  tbz {{w[0-9]+}}, #0,
 %1 = trunc i64 %a to i2
 %2 = trunc i64 %b to i1
; Force fast-isel to fall back to SDAG.
 store i2 %1, i2* %c, align 8
 br i1 %2, label %bb1, label %bb2

bb1:
  ret void

bb2:
  ret void
}
