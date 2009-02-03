; RUN: llvm-as < %s | llc -march=arm -mattr=+v6,+vfp2 | grep fnmuld
; RUN: llvm-as < %s | llc -march=arm -mattr=+v6,+vfp2 -enable-sign-dependent-rounding-fp-math | grep fmul


define double @t1(double %a, double %b) {
entry:
        %tmp2 = sub double -0.000000e+00, %a            ; <double> [#uses=1]
        %tmp4 = mul double %tmp2, %b            ; <double> [#uses=1]
        ret double %tmp4
}

