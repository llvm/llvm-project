; RUN: llc < %s -march=x86

define i1 @T(double %X) {
        %V = fcmp oeq double %X, %X             ; <i1> [#uses=1]
        ret i1 %V
}
