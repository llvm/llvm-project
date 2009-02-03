; Test some floating point casting cases
; RUN: llvm-as %s -o - | opt -instcombine | llvm-dis | notcast
; RUN: llvm-as %s -o - | opt -instcombine | llvm-dis | \
; RUN:   egrep {ret i8 \(-1\)\|\(255\)}

define i8 @test1() {
        %x = fptoui float 2.550000e+02 to i8            ; <i8> [#uses=1]
        ret i8 %x
}

define i8 @test2() {
        %x = fptosi float -1.000000e+00 to i8           ; <i8> [#uses=1]
        ret i8 %x
}

