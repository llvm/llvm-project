; This test makes sure that div instructions are properly eliminated.
; This test is for Integer BitWidth < 64 && BitWidth % 2 != 0.
;
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep div


define i33 @test1(i33 %X) {
    %Y = udiv i33 %X, 4096
    ret i33 %Y
}

define i49 @test2(i49 %X) {
    %tmp.0 = shl i49 4096, 17
    %Y = udiv i49 %X, %tmp.0
    ret i49 %Y
}

define i59 @test3(i59 %X, i1 %C) {
        %V = select i1 %C, i59 1024, i59 4096
        %R = udiv i59 %X, %V
        ret i59 %R
}
