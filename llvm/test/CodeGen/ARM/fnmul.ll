; RUN: llc -mtriple=arm-eabi -mattr=+v6,+vfp2 %s -o - | FileCheck %s

; RUN: llc -mtriple=arm-eabi -mattr=+v6,+vfp2 -enable-sign-dependent-rounding-fp-math %s -o - \
; RUN:  | FileCheck %s -check-prefix CHECK-ROUNDING



define double @t1(double %a, double %b) {
entry:
        %tmp2 = fsub double -0.000000e+00, %a            ; <double> [#uses=1]
        %tmp4 = fmul double %tmp2, %b            ; <double> [#uses=1]
        ret double %tmp4
}

; CHECK: vnmul.f64
; CHECK-ROUNDING: vmul.f64

