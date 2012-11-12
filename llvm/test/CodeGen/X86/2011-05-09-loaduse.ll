; RUN: llc < %s -march=x86 -mcpu=corei7 | FileCheck %s

;CHECK: test
;CHECK-not: pshufd
;CHECK: ret
define float @test(<4 x float>* %A) nounwind {
entry:
  %T = load <4 x float>* %A
  %R = extractelement <4 x float> %T, i32 3
  store <4 x float><float 0.0, float 0.0, float 0.0, float 0.0>, <4 x float>* %A
  ret float %R
}

