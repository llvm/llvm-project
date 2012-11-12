; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s

define ptx_device float @t1_f32(float %x, float %y, float %z) {
; CHECK: fma.rn.f32 %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}};
; CHECK: ret;
  %a = fmul float %x, %y
  %b = fadd float %a, %z
  ret float %b
}

define ptx_device double @t1_f64(double %x, double %y, double %z) {
; CHECK: fma.rn.f64 %fl{{[0-9]+}}, %fl{{[0-9]+}}, %fl{{[0-9]+}}, %fl{{[0-9]+}};
; CHECK: ret;
  %a = fmul double %x, %y
  %b = fadd double %a, %z
  ret double %b
}
