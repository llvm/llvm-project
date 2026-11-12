; This test makes sure that these instructions are properly eliminated.
; This test is for Integer BitWidth >= 64 && BitWidth <= 1024.
;
; RUN: opt < %s -passes=instcombine -S | not grep rem


define i333 @test1(i333 %A) {
    %B = urem i333 %A, 70368744177664
    ret i333 %B
}

define i499 @test2(i499 %A) {
    %B = shl i499 4096, 111
    %Y = urem i499 %A, %B
    ret i499 %Y
}

define i599 @test3(i599 %X, i1 %C) {
	%V = select i1 %C, i599 70368744177664, i599 4096
	%R = urem i599 %X, %V
	ret i599 %R
}
