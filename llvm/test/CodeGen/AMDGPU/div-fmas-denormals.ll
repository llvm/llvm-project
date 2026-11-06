; RUN: llc -mtriple=amdgpu6.00 < %s | FileCheck %s
; RUN: llc -mtriple=amdgpu11.00 < %s | FileCheck %s
; RUN: llc -global-isel -global-isel-abort=1 -mtriple=amdgpu6.00 < %s | FileCheck %s
; RUN: llc -global-isel -global-isel-abort=1 -mtriple=amdgpu11.00 < %s | FileCheck %s

; div.fmas preserves input denormals even when the mode flushes them.
; A false scale flag permits FMA only when the input mode is IEEE.

define float @div_fmas_f32_ieee(float %a, float %b, float %c) #0 {
; CHECK-LABEL: div_fmas_f32_ieee:
; CHECK: v_fma_f32
  %r = call float @llvm.amdgcn.div.fmas.f32(float %a, float %b, float %c, i1 false)
  ret float %r
}

define float @div_fmas_f32_flush_inputs(float %a, float %b, float %c) #1 {
; CHECK-LABEL: div_fmas_f32_flush_inputs:
; CHECK: v_div_fmas_f32
  %r = call float @llvm.amdgcn.div.fmas.f32(float %a, float %b, float %c, i1 false)
  ret float %r
}

define float @div_fmas_f32_flush_outputs(float %a, float %b, float %c) #2 {
; CHECK-LABEL: div_fmas_f32_flush_outputs:
; CHECK: v_fma_f32
  %r = call float @llvm.amdgcn.div.fmas.f32(float %a, float %b, float %c, i1 false)
  ret float %r
}

define float @div_fmas_f32_flush_all(float %a, float %b, float %c) #3 {
; CHECK-LABEL: div_fmas_f32_flush_all:
; CHECK: v_div_fmas_f32
  %r = call float @llvm.amdgcn.div.fmas.f32(float %a, float %b, float %c, i1 false)
  ret float %r
}

define float @div_fmas_f32_dynamic_inputs(float %a, float %b, float %c) #4 {
; CHECK-LABEL: div_fmas_f32_dynamic_inputs:
; CHECK: v_div_fmas_f32
  %r = call float @llvm.amdgcn.div.fmas.f32(float %a, float %b, float %c, i1 false)
  ret float %r
}

define float @div_fmas_f32_override(float %a, float %b, float %c) #5 {
; CHECK-LABEL: div_fmas_f32_override:
; CHECK: v_fma_f32
  %r = call float @llvm.amdgcn.div.fmas.f32(float %a, float %b, float %c, i1 false)
  ret float %r
}

define double @div_fmas_f64_ieee(double %a, double %b, double %c) #0 {
; CHECK-LABEL: div_fmas_f64_ieee:
; CHECK: v_fma_f64
  %r = call double @llvm.amdgcn.div.fmas.f64(double %a, double %b, double %c, i1 false)
  ret double %r
}

define double @div_fmas_f64_flush_inputs(double %a, double %b, double %c) #1 {
; CHECK-LABEL: div_fmas_f64_flush_inputs:
; CHECK: v_div_fmas_f64
  %r = call double @llvm.amdgcn.div.fmas.f64(double %a, double %b, double %c, i1 false)
  ret double %r
}

define double @div_fmas_f64_flush_outputs(double %a, double %b, double %c) #2 {
; CHECK-LABEL: div_fmas_f64_flush_outputs:
; CHECK: v_fma_f64
  %r = call double @llvm.amdgcn.div.fmas.f64(double %a, double %b, double %c, i1 false)
  ret double %r
}

define double @div_fmas_f64_flush_all(double %a, double %b, double %c) #3 {
; CHECK-LABEL: div_fmas_f64_flush_all:
; CHECK: v_div_fmas_f64
  %r = call double @llvm.amdgcn.div.fmas.f64(double %a, double %b, double %c, i1 false)
  ret double %r
}

define double @div_fmas_f64_dynamic_inputs(double %a, double %b, double %c) #4 {
; CHECK-LABEL: div_fmas_f64_dynamic_inputs:
; CHECK: v_div_fmas_f64
  %r = call double @llvm.amdgcn.div.fmas.f64(double %a, double %b, double %c, i1 false)
  ret double %r
}

define double @div_fmas_f64_override(double %a, double %b, double %c) #6 {
; CHECK-LABEL: div_fmas_f64_override:
; CHECK: v_fma_f64
  %r = call double @llvm.amdgcn.div.fmas.f64(double %a, double %b, double %c, i1 false)
  ret double %r
}

define float @div_fmas_f32_true(float %a, float %b, float %c) #0 {
; CHECK-LABEL: div_fmas_f32_true:
; CHECK: v_div_fmas_f32
  %r = call float @llvm.amdgcn.div.fmas.f32(float %a, float %b, float %c, i1 true)
  ret float %r
}

define double @div_fmas_f64_true(double %a, double %b, double %c) #0 {
; CHECK-LABEL: div_fmas_f64_true:
; CHECK: v_div_fmas_f64
  %r = call double @llvm.amdgcn.div.fmas.f64(double %a, double %b, double %c, i1 true)
  ret double %r
}

attributes #0 = { denormal_fpenv(ieee) }
attributes #1 = { denormal_fpenv(ieee|preservesign) }
attributes #2 = { denormal_fpenv(preservesign|ieee) }
attributes #3 = { denormal_fpenv(preservesign) }
attributes #4 = { denormal_fpenv(ieee|dynamic) }
attributes #5 = { denormal_fpenv(preservesign, float: ieee) }
attributes #6 = { denormal_fpenv(ieee, float: preservesign) }
