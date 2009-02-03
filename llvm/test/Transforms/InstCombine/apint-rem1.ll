; This test makes sure that these instructions are properly eliminated.
; This test is for Integer BitWidth < 64 && BitWidth % 2 != 0.
;
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep rem


define i33 @test1(i33 %A) {
    %B = urem i33 %A, 4096
    ret i33 %B
}

define i49 @test2(i49 %A) {
    %B = shl i49 4096, 11
    %Y = urem i49 %A, %B
    ret i49 %Y
}

define i59 @test3(i59 %X, i1 %C) {
	%V = select i1 %C, i59 70368744177664, i59 4096
	%R = urem i59 %X, %V
	ret i59 %R
}
