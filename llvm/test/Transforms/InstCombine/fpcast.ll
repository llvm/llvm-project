; Test some floating point casting cases
; RUN: opt < %s -instcombine -S | FileCheck %s

define i8 @test1() {
        %x = fptoui float 2.550000e+02 to i8            ; <i8> [#uses=1]
        ret i8 %x
; CHECK: ret i8 -1
}

define i8 @test2() {
        %x = fptosi float -1.000000e+00 to i8           ; <i8> [#uses=1]
        ret i8 %x
; CHECK: ret i8 -1
}

