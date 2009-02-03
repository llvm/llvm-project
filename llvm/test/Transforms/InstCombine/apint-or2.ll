; This test makes sure that or instructions are properly eliminated.
; This test is for Integer BitWidth > 64 && BitWidth <= 1024.
;
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep or


define i777 @test0(i777 %X) {
    %Y = or i777 %X, 0
    ret i777 %Y
}

define i117 @test1(i117 %X) {
    %Y = or i117 %X, -1
    ret i117 %Y
} 

define i1023 @test2(i1023 %A) {
    ;; A | ~A == -1
    %NotA = xor i1023 -1, %A
    %B = or i1023 %A, %NotA
    ret i1023 %B
}

define i399 @test3(i399 %V, i399 %M) {
    ;; If we have: ((V + N) & C1) | (V & C2)
    ;; .. and C2 = ~C1 and C2 is 0+1+ and (N & C2) == 0
    ;; replace with V+N.
    %C1 = xor i399 274877906943, -1 ;; C2 = 274877906943
    %N = and i399 %M, 18446742974197923840
    %A = add i399 %V, %N
    %B = and i399 %A, %C1
    %D = and i399 %V, 274877906943
    %R = or i399 %B, %D
    ret i399 %R
}
