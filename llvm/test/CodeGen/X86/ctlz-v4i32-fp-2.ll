; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+sse2,-ssse3 -o - | FileCheck %s

define <4 x i32> @test_v4i32_sse2_zero_undef(<4 x i32> %a) #0 {
; CHECK-LABEL: test_v4i32_sse2_zero_undef:

; zero check
; CHECK-DAG:   pcmpeqd

; FP-based mantissa/exponent steps (order may vary)
; CHECK-DAG:   psrld     $16
; CHECK-DAG:   subps
; CHECK-DAG:   psrld     $23
; CHECK-DAG:   psubd

; merge/select
; CHECK:       pandn
; CHECK:       por

; CHECK-NOT:   pshufb

; CHECK: retq

  %res = call <4 x i32> @llvm.ctlz.v4i32(<4 x i32> %a, i1 true)
  ret <4 x i32> %res
}

declare <4 x i32> @llvm.ctlz.v4i32(<4 x i32>, i1)
attributes #0 = { "optnone" }
