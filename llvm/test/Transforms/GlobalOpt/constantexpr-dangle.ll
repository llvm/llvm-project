; RUN: llvm-as < %s | opt -instcombine -globalopt | llvm-dis | \
; RUN:   grep {internal fastcc float @foo}

define internal float @foo() {
        ret float 0.000000e+00
}

define float @bar() {
        %tmp1 = call float (...)* bitcast (float ()* @foo to float (...)*)( )
        %tmp2 = mul float %tmp1, 1.000000e+01           ; <float> [#uses=1]
        ret float %tmp2
}

