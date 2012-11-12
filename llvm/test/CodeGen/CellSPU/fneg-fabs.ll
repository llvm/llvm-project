; RUN: llc < %s -march=cellspu > %t1.s
; RUN: grep 32768   %t1.s | count 2
; RUN: grep xor     %t1.s | count 4
; RUN: grep and     %t1.s | count 2

target datalayout = "E-p:32:32:128-f64:64:128-f32:32:128-i64:32:128-i32:32:128-i16:16:128-i8:8:128-i1:8:128-a0:0:128-v128:128:128-s0:128:128"
target triple = "spu"

define double @fneg_dp(double %X) {
        %Y = fsub double -0.000000e+00, %X
        ret double %Y
}

define <2 x double> @fneg_dp_vec(<2 x double> %X) {
        %Y = fsub <2 x double> < double -0.0000e+00, double -0.0000e+00 >, %X
        ret <2 x double> %Y
}

define float @fneg_sp(float %X) {
        %Y = fsub float -0.000000e+00, %X
        ret float %Y
}

define <4 x float> @fneg_sp_vec(<4 x float> %X) {
        %Y = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00,
                              float -0.000000e+00, float -0.000000e+00>, %X
        ret <4 x float> %Y
}

declare double @fabs(double)

declare float @fabsf(float)

define double @fabs_dp(double %X) {
        %Y = call double @fabs( double %X ) readnone
        ret double %Y
}

define float @fabs_sp(float %X) {
        %Y = call float @fabsf( float %X ) readnone
        ret float %Y
}
