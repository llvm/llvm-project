; RUN: llc < %s -mtriple=nvptx -mcpu=sm_20 -nvptx-fma-level=1 | FileCheck %s -check-prefix=FMA
; RUN: llc < %s -mtriple=nvptx -mcpu=sm_20 -nvptx-fma-level=0 | FileCheck %s -check-prefix=MUL
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 -nvptx-fma-level=1 | FileCheck %s -check-prefix=FMA
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 -nvptx-fma-level=0 | FileCheck %s -check-prefix=MUL
; RUN: %if ptxas && !ptxas-12.0 %{ llc < %s -mtriple=nvptx -mcpu=sm_20 -nvptx-fma-level=1 | %ptxas-verify %}
; RUN: %if ptxas && !ptxas-12.0 %{ llc < %s -mtriple=nvptx -mcpu=sm_20 -nvptx-fma-level=0 | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 -nvptx-fma-level=1 | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 -nvptx-fma-level=0 | %ptxas-verify %}

define ptx_device float @test_mul_add_f(float %x, float %y, float %z) {
entry:
; FMA: fma.rn.f32
; MUL: mul.rn.f32
; MUL: add.rn.f32
  %a = fmul float %x, %y
  %b = fadd float %a, %z
  ret float %b
}

define ptx_device double @test_mul_add_d(double %x, double %y, double %z) {
entry:
; FMA: fma.rn.f64
; MUL: mul.rn.f64
; MUL: add.rn.f64
  %a = fmul double %x, %y
  %b = fadd double %a, %z
  ret double %b
}
