; RUN: llc -verify-machineinstrs < %s -mattr=+altivec | FileCheck %s

target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Verify that in the 64-bit Linux ABI, vector arguments take up space
; in the parameter save area.

define i64 @callee(i64 %a, <4 x i32> %b, i64 %c, <4 x i32> %d, i64 %e) {
entry:
  ret i64 %e
}
; CHECK-LABEL: callee:
; CHECK: ld 3, 112(1)

define void @caller(i64 %x, <4 x i32> %y) {
entry:
  tail call void @test(i64 %x, <4 x i32> %y, i64 %x, <4 x i32> %y, i64 %x)
  ret void
}
; CHECK-LABEL: caller:
; CHECK: std 3, 112(1)

declare void @test(i64, <4 x i32>, i64, <4 x i32>, i64)

