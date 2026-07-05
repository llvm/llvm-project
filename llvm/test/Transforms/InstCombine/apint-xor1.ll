; This test makes sure that xor instructions are properly eliminated.
; This test is for Integer BitWidth <= 64 && BitWidth % 8 != 0.

; RUN: opt < %s -passes=instcombine -S | not grep "xor "


define i47 @test1(i47 %A, i47 %B) {
        ;; (A & C1)^(B & C2) -> (A & C1)|(B & C2) iff C1&C2 == 0
        %A1 = and i47 %A, 70368744177664
        %B1 = and i47 %B, 70368744177661
        %C1 = xor i47 %A1, %B1
        ret i47 %C1
}

define i15 @test2(i15 %x) {
        %tmp.2 = xor i15 %x, 0
        ret i15 %tmp.2
}

define i23 @test3(i23 %x) {
        %tmp.2 = xor i23 %x, %x
        ret i23 %tmp.2
}

define i37 @test4(i37 %x) {
        ; x ^ ~x == -1
        %NotX = xor i37 -1, %x
        %B = xor i37 %x, %NotX
        ret i37 %B
}

define i7 @test5(i7 %A) {
        ;; (A|B)^B == A & (~B)
        %t1 = or i7 %A, 23
        %r = xor i7 %t1, 23
        ret i7 %r
}

define i7 @test6(i7 %A) {
        %t1 = xor i7 %A, 23
        %r = xor i7 %t1, 23
        ret i7 %r
}

define i47 @test7(i47 %A) {
        ;; (A | C1) ^ C2 -> (A | C1) & ~C2 iff (C1&C2) == C2
        %B1 = or i47 %A,   70368744177663
        %C1 = xor i47 %B1, 703687463
        ret i47 %C1
}
