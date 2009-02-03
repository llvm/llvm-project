; RUN: llvm-as < %s | llc -march=ppc32 | not grep fneg

define double @test1(double %a, double %b, double %c, double %d) {
entry:
        %tmp2 = sub double -0.000000e+00, %c            ; <double> [#uses=1]
        %tmp4 = mul double %tmp2, %d            ; <double> [#uses=1]
        %tmp7 = mul double %a, %b               ; <double> [#uses=1]
        %tmp9 = sub double %tmp7, %tmp4         ; <double> [#uses=1]
        ret double %tmp9
}


