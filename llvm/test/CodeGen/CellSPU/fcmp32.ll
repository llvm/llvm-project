; RUN: llvm-as -o - %s | llc -march=cellspu > %t1.s
; RUN: grep fceq  %t1.s | count 1
; RUN: grep fcmeq %t1.s | count 1

target datalayout = "E-p:32:32:128-f64:64:128-f32:32:128-i64:32:128-i32:32:128-i16:16:128-i8:8:128-i1:8:128-a0:0:128-v128:128:128-s0:128:128"
target triple = "spu"

; Exercise the floating point comparison operators for f32:

declare double @fabs(double)
declare float @fabsf(float)

define i1 @fcmp_eq(float %arg1, float %arg2) {
        %A = fcmp oeq float %arg1,  %arg2
        ret i1 %A
}

define i1 @fcmp_mag_eq(float %arg1, float %arg2) {
        %1 = call float @fabsf(float %arg1)
        %2 = call float @fabsf(float %arg2)
        %3 = fcmp oeq float %1, %2
        ret i1 %3
}
