; RUN: llc < %s -enable-unsafe-fp-math -enable-no-signed-zeros-fp-math -mtriple=x86_64-unknown-unknown -mcpu=corei7 | FileCheck %s

; Make sure that vectors get the same benefits as scalars when using unsafe-fp-math.

; Subtracting zero is free.
define <4 x float> @vec_fsub_zero(<4 x float> %x) {
; CHECK-LABEL: vec_fsub_zero:
; CHECK-NOT: subps
; CHECK-NOT: xorps
; CHECK: retq
  %sub = fsub <4 x float> %x, zeroinitializer
  ret <4 x float> %sub
}

; Negating doesn't require subtraction.
define <4 x float> @vec_fneg(<4 x float> %x) {
; CHECK-LABEL: vec_fneg:
; CHECK: xorps  {{.*}}LCP{{.*}}, %xmm0
; CHECK-NOT: subps
; CHECK-NEXT: retq
  %sub = fsub <4 x float> zeroinitializer, %x
  ret <4 x float> %sub
}
