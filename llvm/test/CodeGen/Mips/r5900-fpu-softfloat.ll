; RUN: llc -mtriple=mips64el -mcpu=r5900 < %s | FileCheck %s
;
; R5900 FPU does not implement IEEE 754 NaN or infinity. Arithmetic operations
; use software libcall fallbacks by default for IEEE correctness. Hardware FPU
; instructions are used when per-instruction nnan+ninf flags guarantee no NaN
; or infinity inputs/outputs.

define float @test_fadd(float %a, float %b) {
; CHECK-LABEL: test_fadd:
; CHECK: jal __addsf3
  %result = fadd float %a, %b
  ret float %result
}

define float @test_fsub(float %a, float %b) {
; CHECK-LABEL: test_fsub:
; CHECK: jal __subsf3
  %result = fsub float %a, %b
  ret float %result
}

define float @test_fmul(float %a, float %b) {
; CHECK-LABEL: test_fmul:
; CHECK: jal __mulsf3
  %result = fmul float %a, %b
  ret float %result
}

define float @test_fdiv(float %a, float %b) {
; CHECK-LABEL: test_fdiv:
; CHECK: jal __divsf3
  %result = fdiv float %a, %b
  ret float %result
}

define float @test_fsqrt(float %a) {
; CHECK-LABEL: test_fsqrt:
; CHECK: jal sqrtf
  %result = call float @llvm.sqrt.f32(float %a)
  ret float %result
}

; Per-instruction nnan+ninf flags select hardware directly.
define float @test_fadd_nnan_ninf(float %a, float %b) {
; CHECK-LABEL: test_fadd_nnan_ninf:
; CHECK: add.s
  %result = fadd nnan ninf float %a, %b
  ret float %result
}

define float @test_fsub_nnan_ninf(float %a, float %b) {
; CHECK-LABEL: test_fsub_nnan_ninf:
; CHECK: sub.s
  %result = fsub nnan ninf float %a, %b
  ret float %result
}

define float @test_fmul_nnan_ninf(float %a, float %b) {
; CHECK-LABEL: test_fmul_nnan_ninf:
; CHECK: mul.s
  %result = fmul nnan ninf float %a, %b
  ret float %result
}

define float @test_fdiv_nnan_ninf(float %a, float %b) {
; CHECK-LABEL: test_fdiv_nnan_ninf:
; CHECK: div.s
  %result = fdiv nnan ninf float %a, %b
  ret float %result
}

define float @test_fsqrt_nnan_ninf(float %a) {
; CHECK-LABEL: test_fsqrt_nnan_ninf:
; CHECK: sqrt.s
  %result = call nnan ninf float @llvm.sqrt.f32(float %a)
  ret float %result
}

declare float @llvm.sqrt.f32(float)
