; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 | not grep r1

define double @test1(double %X) {
        %Y = fptosi double %X to i64            ; <i64> [#uses=1]
        %Z = sitofp i64 %Y to double            ; <double> [#uses=1]
        ret double %Z
}

define float @test2(double %X) {
        %Y = fptosi double %X to i64            ; <i64> [#uses=1]
        %Z = sitofp i64 %Y to float             ; <float> [#uses=1]
        ret float %Z
}

define double @test3(float %X) {
        %Y = fptosi float %X to i64             ; <i64> [#uses=1]
        %Z = sitofp i64 %Y to double            ; <double> [#uses=1]
        ret double %Z
}

define float @test4(float %X) {
        %Y = fptosi float %X to i64             ; <i64> [#uses=1]
        %Z = sitofp i64 %Y to float             ; <float> [#uses=1]
        ret float %Z
}


