; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+sse2,-ssse3 -o - | FileCheck %s

define <4 x i32> @test_v4i32_sse2(<4 x i32> %a) #0 {
; CHECK-LABEL: test_v4i32_sse2:
; CHECK:       # %bb.0:

; Zero test (strict CTLZ needs select)
; CHECK-DAG:   pcmpeqd %xmm{{[0-9]+}}, %xmm{{[0-9]+}}

; Exponent extraction + bias arithmetic (order-free)
; CHECK-DAG:   psrld {{\$}}23, %xmm{{[0-9]+}}
; CHECK-DAG:   psubd %xmm{{[0-9]+}}, %xmm{{[0-9]+}}

; Select/merge (could be por/pandn etc.)
; CHECK:       por %xmm{{[0-9]+}}, %xmm{{[0-9]+}}

; Must NOT use SSSE3 LUT path
; CHECK-NOT:   pshufb

; CHECK:       retq
  %res = call <4 x i32> @llvm.ctlz.v4i32(<4 x i32> %a, i1 false)
  ret <4 x i32> %res
}

declare <4 x i32> @llvm.ctlz.v4i32(<4 x i32>, i1)
attributes #0 = { "optnone" }
