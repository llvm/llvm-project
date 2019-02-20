; RUN: llc -mtriple=i686-unknown-unknown -mattr=+avx512f < %s | FileCheck %s

define float @test_x86_fma_intersection_fmf(float %a, float %b) {
; CHECK-LABEL: test_x86_fma_intersection_fmf:
; CHECK:      # %bb.0:
; CHECK:        vfmadd132ss {{[0-9]+}}(%esp), %xmm0, %xmm0
; CHECK:        retl 
     %tmp8 = fmul fast float %a, %b
     %tmp9 = fadd fast float %tmp8, %b
     %tmp10 = insertelement <4 x float> undef, float  %tmp9, i32 1
     %tmp11 = extractelement <4 x float> %tmp10, i32 1
     ret float %tmp11
}
