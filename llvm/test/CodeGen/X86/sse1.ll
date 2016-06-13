; Tests for SSE1 and below, without SSE2+.
; RUN: llc < %s -mtriple=i386-unknown-unknown -march=x86 -mcpu=pentium3 -O3 | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -march=x86-64 -mattr=-sse2,+sse -O3 | FileCheck %s

; PR7993
;define <4 x i32> @test3(<4 x i16> %a) nounwind {
;  %c = sext <4 x i16> %a to <4 x i32>             ; <<4 x i32>> [#uses=1]
;  ret <4 x i32> %c
;}

; This should not emit shuffles to populate the top 2 elements of the 4-element
; vector that this ends up returning.
; rdar://8368414
define <2 x float> @test4(<2 x float> %A, <2 x float> %B) nounwind {
; CHECK-LABEL: test4:
; CHECK:       # BB#0: # %entry
; CHECK-NEXT:    movaps %xmm0, %xmm2
; CHECK-NEXT:    shufps {{.*#+}} xmm2 = xmm2[1,1,2,3]
; CHECK-NEXT:    addss %xmm1, %xmm0
; CHECK-NEXT:    shufps {{.*#+}} xmm1 = xmm1[1,1,2,3]
; CHECK-NEXT:    subss %xmm1, %xmm2
; CHECK-NEXT:    unpcklps {{.*#+}} xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1]
; CHECK-NEXT:    ret
entry:
  %tmp7 = extractelement <2 x float> %A, i32 0
  %tmp5 = extractelement <2 x float> %A, i32 1
  %tmp3 = extractelement <2 x float> %B, i32 0
  %tmp1 = extractelement <2 x float> %B, i32 1
  %add.r = fadd float %tmp7, %tmp3
  %add.i = fsub float %tmp5, %tmp1
  %tmp11 = insertelement <2 x float> undef, float %add.r, i32 0
  %tmp9 = insertelement <2 x float> %tmp11, float %add.i, i32 1
  ret <2 x float> %tmp9
}

; We used to get stuck in type legalization for this example when lowering the
; vselect. With SSE1 v4f32 is a legal type but v4i1 (or any vector integer type)
; is not. We used to ping pong between splitting the vselect for the v4i
; condition operand and widening the resulting vselect for the v4f32 result.
; PR18036

define <4 x float> @vselect(<4 x float>*%p, <4 x i32> %q) {
; CHECK-LABEL: vselect:
; CHECK:         ret
entry:
  %a1 = icmp eq <4 x i32> %q, zeroinitializer
  %a14 = select <4 x i1> %a1, <4 x float> <float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+0> , <4 x float> zeroinitializer
  ret <4 x float> %a14
}

; v4i32 isn't legal for SSE1, but this should be cmpps.

define <4 x float> @PR28044(<4 x float> %a0, <4 x float> %a1) nounwind {
; CHECK-LABEL: PR28044:
; CHECK:       # BB#0:
; CHECK-NEXT:    cmpeqps %xmm1, %xmm0
; CHECK-NEXT:    ret
;
  %cmp = fcmp oeq <4 x float> %a0, %a1
  %sext = sext <4 x i1> %cmp to <4 x i32>
  %res = bitcast <4 x i32> %sext to <4 x float>
  ret <4 x float> %res
}

