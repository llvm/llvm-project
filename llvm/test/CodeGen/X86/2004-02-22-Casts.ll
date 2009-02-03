; RUN: llvm-as < %s | llc -march=x86
define i1 @test1(double %X) {
        %V = fcmp one double %X, 0.000000e+00           ; <i1> [#uses=1]
        ret i1 %V
}

define double @test2(i64 %X) {
        %V = uitofp i64 %X to double            ; <double> [#uses=1]
        ret double %V
}


