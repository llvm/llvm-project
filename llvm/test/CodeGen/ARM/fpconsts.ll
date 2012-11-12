; RUN: llc < %s -march=arm -mattr=+vfp3 | FileCheck %s

define float @t1(float %x) nounwind readnone optsize {
entry:
; CHECK: t1:
; CHECK: vmov.f32 s{{.*}}, #4.000000e+00
  %0 = fadd float %x, 4.000000e+00
  ret float %0
}

define double @t2(double %x) nounwind readnone optsize {
entry:
; CHECK: t2:
; CHECK: vmov.f64 d{{.*}}, #3.000000e+00
  %0 = fadd double %x, 3.000000e+00
  ret double %0
}

define double @t3(double %x) nounwind readnone optsize {
entry:
; CHECK: t3:
; CHECK: vmov.f64 d{{.*}}, #-1.300000e+01
  %0 = fmul double %x, -1.300000e+01
  ret double %0
}

define float @t4(float %x) nounwind readnone optsize {
entry:
; CHECK: t4:
; CHECK: vmov.f32 s{{.*}}, #-2.400000e+01
  %0 = fmul float %x, -2.400000e+01
  ret float %0
}
