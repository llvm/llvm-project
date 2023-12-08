; The output formater prints out 1.0e100 as Inf!
;
; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | grep 0x7FF0000000000000
; RUN: verify-uselistorder %s

define float @test() {
        %tmp = fmul float 0x7FF0000000000000, 1.000000e+01               ; <float> [#uses=1]
        ret float %tmp
}

