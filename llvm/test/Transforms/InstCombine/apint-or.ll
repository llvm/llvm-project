; RUN: opt < %s -instcombine -S | FileCheck %s

; These tests are for Integer BitWidth <= 64 && BitWidth % 2 != 0.
define i23 @test1(i23 %A) {
    ;; A | ~A == -1
    %NotA = xor i23 -1, %A
    %B = or i23 %A, %NotA
    ret i23 %B
; CHECK-LABEL: @test1
; CHECK-NEXT: ret i23 -1
}

define i39 @test2(i39 %V, i39 %M) {
    ;; If we have: ((V + N) & C1) | (V & C2)
    ;; .. and C2 = ~C1 and C2 is 0+1+ and (N & C2) == 0
    ;; replace with V+N.
    %C1 = xor i39 274877906943, -1 ;; C2 = 274877906943
    %N = and i39 %M, 274877906944
    %A = add i39 %V, %N
    %B = and i39 %A, %C1
    %D = and i39 %V, 274877906943
    %R = or i39 %B, %D
    ret i39 %R
; CHECK-LABEL: @test2
; CHECK-NEXT: %N = and i39 %M, -274877906944
; CHECK-NEXT: %A = add i39 %N, %V
; CHECK-NEXT: ret i39 %A
}

; These tests are for Integer BitWidth > 64 && BitWidth <= 1024.
define i1023 @test4(i1023 %A) {
    ;; A | ~A == -1
    %NotA = xor i1023 -1, %A
    %B = or i1023 %A, %NotA
    ret i1023 %B
; CHECK-LABEL: @test4
; CHECK-NEXT: ret i1023 -1
}

define i399 @test5(i399 %V, i399 %M) {
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
; CHECK-LABEL: @test5
; CHECK-NEXT: %N = and i399 %M, 18446742974197923840
; CHECK-NEXT: %A = add i399 %N, %V
; CHECK-NEXT: ret i399 %A
}

