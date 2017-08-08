; RUN: opt < %s -S -early-cse | FileCheck %s
; RUN: opt < %s -S -basicaa -early-cse-memssa | FileCheck %s

; Ensure we don't simplify away additions vectors of +0.0's (same as scalars).
define <4 x float> @fV( <4 x float> %a) {
       ; CHECK: %b = fadd <4 x float> %a, zeroinitializer
       %b = fadd  <4 x float> %a, <float 0.0,float 0.0,float 0.0,float 0.0>
       ret <4 x float> %b
}

define <4 x float> @fW( <4 x float> %a) {
       ; CHECK: ret <4 x float> %a
       %b = fadd  <4 x float> %a, <float -0.0,float -0.0,float -0.0,float -0.0>
       ret <4 x float> %b
}
