; RUN: llvm-as < %s | llc -march=x86

define double @test(double %d) {
        %X = select i1 false, double %d, double %d              ; <double> [#uses=1]
        ret double %X
}

