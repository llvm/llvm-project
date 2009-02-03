; This test makes sure that these instructions are properly eliminated.
;
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep -v xor | not grep {or }

; PR1738
define i1 @test1(double %X, double %Y) {
        %tmp9 = fcmp uno double %X, 0.000000e+00                ; <i1> [#uses=1]
        %tmp13 = fcmp uno double %Y, 0.000000e+00               ; <i1> [#uses=1]
        %bothcond = or i1 %tmp13, %tmp9         ; <i1> [#uses=1]
        ret i1 %bothcond
}

