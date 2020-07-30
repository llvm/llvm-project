; RUN: opt < %s -S -early-cse | FileCheck %s
; RUN: opt < %s -S -basic-aa -early-cse-memssa | FileCheck %s

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

; CSE unary fnegs.
define void @fX(<4 x float> *%p, <4 x float> %a) {
       ; CHECK: %x = fneg <4 x float> %a
       ; CHECK-NEXT: store volatile <4 x float> %x, <4 x float>* %p
       ; CHECK-NEXT: store volatile <4 x float> %x, <4 x float>* %p
       %x = fneg <4 x float> %a
       %y = fneg <4 x float> %a
       store volatile <4 x float> %x, <4 x float>* %p
       store volatile <4 x float> %y, <4 x float>* %p
       ret void
}
