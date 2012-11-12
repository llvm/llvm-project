; RUN: opt < %s -simplify-libcalls -S | FileCheck %s
; rdar://7251832

; SimplifyLibcalls should optimize pow(x, 0.5) to sqrt plus code to handle
; special cases. The readonly attribute on the call should be preserved.

; CHECK: define float @foo(float %x) nounwind {
; CHECK:   %sqrtf = call float @sqrtf(float %x) nounwind readonly
; CHECK:   %fabsf = call float @fabsf(float %sqrtf) nounwind readonly
; CHECK:   %1 = fcmp oeq float %x, 0xFFF0000000000000
; CHECK:   %retval = select i1 %1, float 0x7FF0000000000000, float %fabsf
; CHECK:   ret float %retval

define float @foo(float %x) nounwind {
  %retval = call float @powf(float %x, float 0.5)
  ret float %retval
}

; CHECK: define double @doo(double %x) nounwind {
; CHECK:   %sqrt = call double @sqrt(double %x) nounwind readonly
; CHECK:   %fabs = call double @fabs(double %sqrt) nounwind readonly
; CHECK:   %1 = fcmp oeq double %x, 0xFFF0000000000000
; CHECK:   %retval = select i1 %1, double 0x7FF0000000000000, double %fabs
; CHECK:   ret double %retval
; CHECK: }

define double @doo(double %x) nounwind {
  %retval = call double @pow(double %x, double 0.5)
  ret double %retval
}

declare float @powf(float, float) nounwind readonly
declare double @pow(double, double) nounwind readonly
