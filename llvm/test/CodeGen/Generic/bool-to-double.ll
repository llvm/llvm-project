; RUN: llvm-as < %s | llc
define double @test(i1 %X) {
        %Y = uitofp i1 %X to double             ; <double> [#uses=1]
        ret double %Y
}

