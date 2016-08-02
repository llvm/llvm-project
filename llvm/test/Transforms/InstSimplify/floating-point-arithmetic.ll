; RUN: opt < %s -instsimplify -S | FileCheck %s

; fsub 0, (fsub 0, X) ==> X
; CHECK-LABEL: @fsub_0_0_x(
define float @fsub_0_0_x(float %a) {
  %t1 = fsub float -0.0, %a
  %ret = fsub float -0.0, %t1

; CHECK: ret float %a
  ret float %ret
}

; fsub X, 0 ==> X
; CHECK-LABEL: @fsub_x_0(
define float @fsub_x_0(float %a) {
  %ret = fsub float %a, 0.0
; CHECK: ret float %a
  ret float %ret
}

; fadd X, -0 ==> X
; CHECK-LABEL: @fadd_x_n0(
define float @fadd_x_n0(float %a) {
  %ret = fadd float %a, -0.0
; CHECK: ret float %a
  ret float %ret
}

; fmul X, 1.0 ==> X
; CHECK-LABEL: @fmul_X_1(
define double @fmul_X_1(double %a) {
  %b = fmul double 1.000000e+00, %a                ; <double> [#uses=1]
  ; CHECK: ret double %a
  ret double %b
}

; We can't optimize away the fadd in this test because the input
; value to the function and subsequently to the fadd may be -0.0. 
; In that one special case, the result of the fadd should be +0.0
; rather than the first parameter of the fadd.

; Fragile test warning: We need 6 sqrt calls to trigger the bug 
; because the internal logic has a magic recursion limit of 6. 
; This is presented without any explanation or ability to customize.

declare float @sqrtf(float)

define float @PR22688(float %x) {
  %1 = call float @sqrtf(float %x)
  %2 = call float @sqrtf(float %1)
  %3 = call float @sqrtf(float %2)
  %4 = call float @sqrtf(float %3)
  %5 = call float @sqrtf(float %4)
  %6 = call float @sqrtf(float %5)
  %7 = fadd float %6, 0.0
  ret float %7

; CHECK-LABEL: @PR22688(
; CHECK: fadd float %6, 0.0
}

