; This test makes sure that mul instructions are properly eliminated.
; This test is for Integer BitWidth < 64 && BitWidth % 2 != 0.
;

; RUN: opt < %s -instcombine -S | not grep mul


define i17 @test1(i17 %X) {
    %Y = mul i17 %X, 1024
    ret i17 %Y
} 
