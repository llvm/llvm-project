; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+ssse3 -o - | FileCheck %s

; This verifies that **with SSSE3 enabled**, we use the LUT-based `pshufb`
; implementation and *not* the floating-point exponent trick.

define <4 x i32> @test_v4i32_ssse3(<4 x i32> %a) {
; CHECK-LABEL: test_v4i32_ssse3:
; CHECK:       # %bb.0:

; Must use SSSE3 table LUT:
; CHECK:       pshufb

; Must NOT use FP exponent trick:
; CHECK-NOT:   cvtdq2ps
; CHECK-NOT:   psrld $23
; CHECK-NOT:   psubd

; CHECK:       retq
  %res = call <4 x i32> @llvm.ctlz.v4i32(<4 x i32> %a, i1 false)
  ret <4 x i32> %res
}

declare <4 x i32> @llvm.ctlz.v4i32(<4 x i32>, i1)
