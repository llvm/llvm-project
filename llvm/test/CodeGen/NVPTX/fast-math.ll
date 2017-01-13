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

attributes #0 = { "unsafe-fp-math" = "true" }
attributes #1 = { "nvptx-f32ftz" = "true" }
