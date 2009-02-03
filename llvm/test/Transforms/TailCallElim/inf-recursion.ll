; RUN: llvm-as < %s | opt -tailcallelim | llvm-dis | grep call
; Don't turn this into an infinite loop, this is probably the implementation
; of fabs and we expect the codegen to lower fabs.

define double @fabs(double %f) {
entry:
        %tmp2 = call double @fabs( double %f )          ; <double> [#uses=1]
        ret double %tmp2
}

