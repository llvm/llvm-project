; This test makes sure that div instructions are properly eliminated.
; This test is for Integer BitWidth >= 64 && BitWidth <= 1024.
;
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep div


define i333 @test1(i333 %X) {
    %Y = udiv i333 %X, 70368744177664
    ret i333 %Y
}

define i499 @test2(i499 %X) {
    %tmp.0 = shl i499 4096, 197
    %Y = udiv i499 %X, %tmp.0
    ret i499 %Y
}

define i599 @test3(i599 %X, i1 %C) {
        %V = select i1 %C, i599 70368744177664, i599 4096
        %R = udiv i599 %X, %V
        ret i599 %R
}
