; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 -fp-contract=fast | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 -fp-contract=fast | %ptxas-verify %}

target triple = "nvptx64-unknown-cuda"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

; Make sure we can properly differentiate between single-precision and
; double-precision FP literals.

; CHECK: myaddf
; CHECK: add.f32 %f{{[0-9]+}}, %f{{[0-9]+}}, 0f3F800000
define float @myaddf(float %a) {
  %ret = fadd float %a, 1.0
  ret float %ret
}

; CHECK: myaddd
; CHECK: add.f64 %fd{{[0-9]+}}, %fd{{[0-9]+}}, 0d3FF0000000000000
define double @myaddd(double %a) {
  %ret = fadd double %a, 1.0
  ret double %ret
}
