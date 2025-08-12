; The output formatter prints out 1.0e100 as Inf!
;
; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: fmul float 0x7FF0000000000000, 1.000000e+01
define float @test() {
        %tmp = fmul float 0x7FF0000000000000, 1.000000e+01               ; <float> [#uses=1]
        ret float %tmp
}

