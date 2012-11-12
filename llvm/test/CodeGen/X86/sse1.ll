; Tests for SSE1 and below, without SSE2+.
; RUN: llc < %s -march=x86 -mcpu=pentium3 -O3 | FileCheck %s
; RUN: llc < %s -march=x86-64 -mattr=-sse2,+sse -O3 | FileCheck %s

define <8 x i16> @test1(<8 x i32> %a) nounwind {
; CHECK: test1
  ret <8 x i16> zeroinitializer
}

define <8 x i16> @test2(<8 x i32> %a) nounwind {
; CHECK: test2
  %c = trunc <8 x i32> %a to <8 x i16>            ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %c
}

; PR7993
;define <4 x i32> @test3(<4 x i16> %a) nounwind {
;  %c = sext <4 x i16> %a to <4 x i32>             ; <<4 x i32>> [#uses=1]
;  ret <4 x i32> %c
;}

; This should not emit shuffles to populate the top 2 elements of the 4-element
; vector that this ends up returning.
; rdar://8368414
define <2 x float> @test4(<2 x float> %A, <2 x float> %B) nounwind {
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
; CHECK: test4:
; CHECK-NOT: shufps	$16
; CHECK: shufps	$1, 
; CHECK-NOT: shufps	$16
; CHECK: shufps	$1, 
; CHECK-NOT: shufps	$16
; CHECK: unpcklps
; CHECK-NOT: shufps	$16
; CHECK: ret
}
