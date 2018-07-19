; RUN: opt -reassociate %s -S -o - | FileCheck %s

; Reassociate pass used to crash on these example


define float @undef1() {
wrapper_entry:
; CHECK-LABEL: @undef1
; CHECK: ret float fadd (float undef, float fadd (float undef, float fadd (float fsub (float -0.000000e+00, float undef), float fsub (float -0.000000e+00, float undef))))
  %0 = fadd fast float undef, undef
  %1 = fsub fast float undef, %0
  %2 = fadd fast float undef, %1
  ret float %2
}

define void @undef2() {
wrapper_entry:
; CHECK-LABEL: @undef2
; CHECK: unreachable
  %0 = fadd fast float undef, undef
  %1 = fadd fast float %0, 1.000000e+00
  %2 = fsub fast float %0, %1
  %3 = fmul fast float %2, 2.000000e+00
  unreachable
}
