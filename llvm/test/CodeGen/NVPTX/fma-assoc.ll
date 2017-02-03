; RUN: llc < %s -march=nvptx -mcpu=sm_20 -fp-contract=fast | FileCheck %s -check-prefix=CHECK
; RUN: llc < %s -march=nvptx -mcpu=sm_20 -fp-contract=fast -enable-unsafe-fp-math | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-UNSAFE

define ptx_device float @t1_f32(float %x, float %y, float %z,
                                float %u, float %v) {
; CHECK-UNSAFE: fma.rn.f32 %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}};
; CHECK-UNSAFE: fma.rn.f32 %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}};
; CHECK: ret;
  %a = fmul float %x, %y
  %b = fmul float %u, %v
  %c = fadd float %a, %b
  %d = fadd float %c, %z
  ret float %d
}

define ptx_device double @t1_f64(double %x, double %y, double %z,
                                 double %u, double %v) {
; CHECK-UNSAFE: fma.rn.f64 %fd{{[0-9]+}}, %fd{{[0-9]+}}, %fd{{[0-9]+}}, %fd{{[0-9]+}};
; CHECK-UNSAFE: fma.rn.f64 %fd{{[0-9]+}}, %fd{{[0-9]+}}, %fd{{[0-9]+}}, %fd{{[0-9]+}};
; CHECK: ret;
  %a = fmul double %x, %y
  %b = fmul double %u, %v
  %c = fadd double %a, %b
  %d = fadd double %c, %z
  ret double %d
}

define double @two_choices(double %val1, double %val2) {
; CHECK-LABEL: two_choices(
; CHECK: mul.f64
; CHECK-NOT: mul.f64
; CHECK: fma.rn.f64
  %1 = fmul double %val1, %val2
  %2 = fmul double %1, %1
  %3 = fadd double %1, %2

  ret double %3
}

