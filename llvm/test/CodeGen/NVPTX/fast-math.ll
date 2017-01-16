; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s

declare float @llvm.nvvm.sqrt.f(float)

; CHECK-LABEL: sqrt_div
; CHECK: sqrt.rn.f32
; CHECK: div.rn.f32
define float @sqrt_div(float %a, float %b) {
  %t1 = tail call float @llvm.nvvm.sqrt.f(float %a)
  %t2 = fdiv float %t1, %b
  ret float %t2
}

; CHECK-LABEL: sqrt_div_fast
; CHECK: sqrt.approx.f32
; CHECK: div.approx.f32
define float @sqrt_div_fast(float %a, float %b) #0 {
  %t1 = tail call float @llvm.nvvm.sqrt.f(float %a)
  %t2 = fdiv float %t1, %b
  ret float %t2
}

; CHECK-LABEL: fadd
; CHECK: add.rn.f32
define float @fadd(float %a, float %b) {
  %t1 = fadd float %a, %b
  ret float %t1
}

; CHECK-LABEL: fadd_ftz
; CHECK: add.rn.ftz.f32
define float @fadd_ftz(float %a, float %b) #1 {
  %t1 = fadd float %a, %b
  ret float %t1
}

declare float @llvm.sin.f32(float)
declare float @llvm.cos.f32(float)

; CHECK-LABEL: fsin_approx
; CHECK:       sin.approx.f32
define float @fsin_approx(float %a) #0 {
  %r = tail call float @llvm.sin.f32(float %a)
  ret float %r
}

; CHECK-LABEL: fcos_approx
; CHECK:       cos.approx.f32
define float @fcos_approx(float %a) #0 {
  %r = tail call float @llvm.cos.f32(float %a)
  ret float %r
}

attributes #0 = { "unsafe-fp-math" = "true" }
attributes #1 = { "nvptx-f32ftz" = "true" }
