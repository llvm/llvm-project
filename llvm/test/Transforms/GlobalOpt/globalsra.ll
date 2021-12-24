; RUN: opt < %s -passes=globalopt -S | FileCheck %s
; CHECK-NOT: global
target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

@G = internal global { i32, float, { double } } {
    i32 1, 
    float 1.000000e+00, 
    { double } { double 1.727000e+01 } }                ; <{ i32, float, { double } }*> [#uses=3]

define void @onlystore() {
        store i32 123, i32* getelementptr ({ i32, float, { double } }, { i32, float, { double } }* @G, i32 0, i32 0)
        ret void
}

define float @storeinit() {
        store float 1.000000e+00, float* getelementptr ({ i32, float, { double } }, { i32, float, { double } }* @G, i32 0, i32 1)
        %X = load float, float* getelementptr ({ i32, float, { double } }, { i32, float, { double } }* @G, i32 0, i32 1)           ; <float> [#uses=1]
        ret float %X
}

define double @constantize() {
        %X = load double, double* getelementptr ({ i32, float, { double } }, { i32, float, { double } }* @G, i32 0, i32 2, i32 0)           ; <double> [#uses=1]
        ret double %X
}

@G2 = internal constant { i32, float, { double } } {
    i32 1, 
    float 1.000000e+00, 
    { double } { double 1.727000e+01 } }                ; <{ i32, float, { double } }*> [#uses=3]

define void @onlystore2() {
        store i32 123, i32* getelementptr ({ i32, float, { double } }, { i32, float, { double } }* @G2, i32 0, i32 0)
        ret void
}

define float @storeinit2() {
        store float 1.000000e+00, float* getelementptr ({ i32, float, { double } }, { i32, float, { double } }* @G2, i32 0, i32 1)
        %X = load float, float* getelementptr ({ i32, float, { double } }, { i32, float, { double } }* @G2, i32 0, i32 1)           ; <float> [#uses=1]
        ret float %X
}

define double @constantize2() {
        %X = load double, double* getelementptr ({ i32, float, { double } }, { i32, float, { double } }* @G2, i32 0, i32 2, i32 0)           ; <double> [#uses=1]
        ret double %X
}
