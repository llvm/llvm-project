; RUN: llc < %s -mtriple=i686-- -mattr=+sse2 -enable-unsafe-fp-math | \
; RUN:  grep -v sp | grep xorps | count 8

; Don't fold the incoming stack arguments into the xorps instructions used
; to do floating-point negations, because the arguments aren't vectors
; and aren't vector-aligned.

define void @foo(float* %p, float* %q, float %s, float %y) {
  %ss = fsub float -0.0, %s
  %yy = fsub float -0.0, %y
  store float %ss, float* %p
  store float %yy, float* %q
  ret void
}

define void @foo_unary_fneg_x_y(float* %p, float* %q, float %s, float %y) {
  %ss = fneg float %s
  %yy = fneg float %y
  store float %ss, float* %p
  store float %yy, float* %q
  ret void
}

define void @foo_unary_fneg_x(float* %p, float* %q, float %s, float %y) {
  %ss = fneg float %s
  %yy = fsub float -0.0, %y
  store float %ss, float* %p
  store float %yy, float* %q
  ret void
}

define void @foo_unary_fneg_y(float* %p, float* %q, float %s, float %y) {
  %ss = fsub float -0.0, %s
  %yy = fneg float %y
  store float %ss, float* %p
  store float %yy, float* %q
  ret void
}
