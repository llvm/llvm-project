; RUN: llc -mtriple=arm-eabi -mattr=+v6,+vfp2 %s -o -                        | FileCheck %s -check-prefix STRICT

; RUN: llc -mtriple=arm-eabi -mattr=+v6,+vfp2 -enable-unsafe-fp-math %s -o - | FileCheck %s -check-prefix UNSAFE

define double @t1(double %a, double %b) {
; STRICT:    vnmul.f64
;
; UNSAFE:    vnmul.f64
entry:
        %tmp2 = fsub double -0.000000e+00, %a            ; <double> [#uses=1]
        %tmp4 = fmul double %tmp2, %b            ; <double> [#uses=1]
        ret double %tmp4
}


