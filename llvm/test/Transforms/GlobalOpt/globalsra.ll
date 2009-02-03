; RUN: llvm-as < %s | opt -globalopt | llvm-dis | not grep global

@G = internal global { i32, float, { double } } {
    i32 1, 
    float 1.000000e+00, 
    { double } { double 1.727000e+01 } }                ; <{ i32, float, { double } }*> [#uses=3]

define void @onlystore() {
        store i32 123, i32* getelementptr ({ i32, float, { double } }* @G, i32 0, i32 0)
        ret void
}

define float @storeinit() {
        store float 1.000000e+00, float* getelementptr ({ i32, float, { double } }* @G, i32 0, i32 1)
        %X = load float* getelementptr ({ i32, float, { double } }* @G, i32 0, i32 1)           ; <float> [#uses=1]
        ret float %X
}

define double @constantize() {
        %X = load double* getelementptr ({ i32, float, { double } }* @G, i32 0, i32 2, i32 0)           ; <double> [#uses=1]
        ret double %X
}

