; RUN: llc < %s -march=nvptx -mcpu=sm_20 -fp-contract=fast -verify-machineinstrs | FileCheck %s

declare float @dummy_f32(float, float) #0
declare double @dummy_f64(double, double) #0

define ptx_device float @t1_f32(float %x, float %y, float %z) {
; CHECK: fma.rn.f32 %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}};
; CHECK: ret;
  %a = fmul float %x, %y
  %b = fadd float %a, %z
  ret float %b
}

define ptx_device float @t2_f32(float %x, float %y, float %z, float %w) {
; CHECK: fma.rn.f32 %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}};
; CHECK: fma.rn.f32 %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}};
; CHECK: ret;
  %a = fmul float %x, %y
  %b = fadd float %a, %z
  %c = fadd float %a, %w
  %d = call float @dummy_f32(float %b, float %c)
  ret float %d
}

define ptx_device double @t1_f64(double %x, double %y, double %z) {
; CHECK: fma.rn.f64 %fd{{[0-9]+}}, %fd{{[0-9]+}}, %fd{{[0-9]+}}, %fd{{[0-9]+}};
; CHECK: ret;
  %a = fmul double %x, %y
  %b = fadd double %a, %z
  ret double %b
}

define ptx_device double @t2_f64(double %x, double %y, double %z, double %w) {
; CHECK: fma.rn.f64 %fd{{[0-9]+}}, %fd{{[0-9]+}}, %fd{{[0-9]+}}, %fd{{[0-9]+}};
; CHECK: fma.rn.f64 %fd{{[0-9]+}}, %fd{{[0-9]+}}, %fd{{[0-9]+}}, %fd{{[0-9]+}};
; CHECK: ret;
  %a = fmul double %x, %y
  %b = fadd double %a, %z
  %c = fadd double %a, %w
  %d = call double @dummy_f64(double %b, double %c)
  ret double %d
}
