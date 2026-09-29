; RUN: llc < %s -mtriple=i686--

define double @test(double %d) {
        %X = select i1 false, double %d, double %d              ; <double> [#uses=1]
        ret double %X
}

