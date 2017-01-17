; RUN: llc < %s -mtriple=arm64-apple-ios | FileCheck %s

; CHECK-LABEL: _test:
; CHECK-DAG:  fmov.2d v0, #2.00000000
; CHECK-DAG: and [[MASK_IDX:x[0-9]+]], x1, #0x3
; CHECK-DAG:  mov  x9, sp
; CHECK-DAG:  str  q0, [sp], #16
; CHECK-DAG:  bfi [[PTR:x[0-9]+]], [[MASK_IDX]], #2, #2
; CHECK:  ldr s0, {{\[}}[[PTR]]{{\]}}
; CHECK:  str  s0, [x0]

define void @test(float * %p1, i32 %v1) {
entry:
  %v2 = extractelement <3 x float> <float 0.000000e+00, float 2.000000e+00, float 0.000000e+00>, i32 %v1
  store float %v2, float* %p1, align 4
  ret void
}

; CHECK-LABEL: _test2
; CHECK: movi.16b  v0, #63
; CHECK-DAG: and [[MASK_IDX:x[0-9]+]], x1, #0x3
; CHECK-DAG: str  q0, [sp], #16
; CHECK-DAG: mov  x9, sp
; CHECK-DAG:  bfi [[PTR:x[0-9]+]], [[MASK_IDX]], #2, #2
; CHECK: ldr s0, {{\[}}[[PTR]]{{\]}}
; CHECK: str  s0, [x0]

define void @test2(float * %p1, i32 %v1) {
entry:
  %v2 = extractelement <3 x float> <float 0.7470588088035583, float 0.7470588088035583, float 0.7470588088035583>, i32 %v1
  store float %v2, float* %p1, align 4
  ret void
}
