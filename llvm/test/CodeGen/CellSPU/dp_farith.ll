; RUN: llvm-as -o - %s | llc -march=cellspu > %t1.s
; RUN: grep dfa    %t1.s | count 2
; RUN: grep dfs    %t1.s | count 2
; RUN: grep dfm    %t1.s | count 6
; RUN: grep dfma   %t1.s | count 2
; RUN: grep dfms   %t1.s | count 2
; RUN: grep dfnms  %t1.s | count 4
;
; This file includes double precision floating point arithmetic instructions
target datalayout = "E-p:32:32:128-f64:64:128-f32:32:128-i64:32:128-i32:32:128-i16:16:128-i8:8:128-i1:8:128-a0:0:128-v128:128:128-s0:128:128"
target triple = "spu"

define double @fadd(double %arg1, double %arg2) {
        %A = add double %arg1, %arg2
        ret double %A
}

define <2 x double> @fadd_vec(<2 x double> %arg1, <2 x double> %arg2) {
        %A = add <2 x double> %arg1, %arg2
        ret <2 x double> %A
}

define double @fsub(double %arg1, double %arg2) {
        %A = sub double %arg1,  %arg2
        ret double %A
}

define <2 x double> @fsub_vec(<2 x double> %arg1, <2 x double> %arg2) {
        %A = sub <2 x double> %arg1,  %arg2
        ret <2 x double> %A
}

define double @fmul(double %arg1, double %arg2) {
        %A = mul double %arg1,  %arg2
        ret double %A
}

define <2 x double> @fmul_vec(<2 x double> %arg1, <2 x double> %arg2) {
        %A = mul <2 x double> %arg1,  %arg2
        ret <2 x double> %A
}

define double @fma(double %arg1, double %arg2, double %arg3) {
        %A = mul double %arg1,  %arg2
        %B = add double %A, %arg3
        ret double %B
}

define <2 x double> @fma_vec(<2 x double> %arg1, <2 x double> %arg2, <2 x double> %arg3) {
        %A = mul <2 x double> %arg1,  %arg2
        %B = add <2 x double> %A, %arg3
        ret <2 x double> %B
}

define double @fms(double %arg1, double %arg2, double %arg3) {
        %A = mul double %arg1,  %arg2
        %B = sub double %A, %arg3
        ret double %B
}

define <2 x double> @fms_vec(<2 x double> %arg1, <2 x double> %arg2, <2 x double> %arg3) {
        %A = mul <2 x double> %arg1,  %arg2
        %B = sub <2 x double> %A, %arg3
        ret <2 x double> %B
}

; - (a * b - c)
define double @d_fnms_1(double %arg1, double %arg2, double %arg3) {
        %A = mul double %arg1,  %arg2
        %B = sub double %A, %arg3
        %C = sub double -0.000000e+00, %B               ; <double> [#uses=1]
        ret double %C
}

; Annother way of getting fnms
; - ( a * b ) + c => c - (a * b)
define double @d_fnms_2(double %arg1, double %arg2, double %arg3) {
        %A = mul double %arg1,  %arg2
        %B = sub double %arg3, %A
        ret double %B
}

; FNMS: - (a * b - c) => c - (a * b)
define <2 x double> @d_fnms_vec_1(<2 x double> %arg1, <2 x double> %arg2, <2 x double> %arg3) {
        %A = mul <2 x double> %arg1,  %arg2
        %B = sub <2 x double> %arg3, %A ;
        ret <2 x double> %B
}

; Another way to get fnms using a constant vector
; - ( a * b - c)
define <2 x double> @d_fnms_vec_2(<2 x double> %arg1, <2 x double> %arg2, <2 x double> %arg3) {
        %A = mul <2 x double> %arg1,  %arg2     ; <<2 x double>> [#uses=1]
        %B = sub <2 x double> %A, %arg3 ; <<2 x double>> [#uses=1]
        %C = sub <2 x double> < double -0.00000e+00, double -0.00000e+00 >, %B
        ret <2 x double> %C
}

;define double @fdiv_1(double %arg1, double %arg2) {
;       %A = fdiv double %arg1,  %arg2  ; <double> [#uses=1]
;       ret double %A
;}
