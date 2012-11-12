; This test makes sure that and instructions are properly eliminated.
; This test is for Integer BitWidth <= 64 && BitWidth % 8 != 0.

; RUN: opt < %s -instcombine -S | not grep "and "
; END.

define i39 @test0(i39 %A) {
        %B = and i39 %A, 0 ; zero result
        ret i39 %B
}

define i47 @test1(i47 %A, i47 %B) {
        ;; (~A & ~B) == (~(A | B)) - De Morgan's Law
        %NotA = xor i47 %A, -1
        %NotB = xor i47 %B, -1
        %C1 = and i47 %NotA, %NotB
        ret i47 %C1
}

define i15 @test2(i15 %x) {
        %tmp.2 = and i15 %x, -1 ; noop
        ret i15 %tmp.2
}

define i23 @test3(i23 %x) {
        %tmp.0 = and i23 %x, 127
        %tmp.2 = and i23 %tmp.0, 128
        ret i23 %tmp.2
}

define i1 @test4(i37 %x) {
        %A = and i37 %x, -2147483648
        %B = icmp ne i37 %A, 0
        ret i1 %B
}

define i7 @test5(i7 %A, i7* %P) {
        %B = or i7 %A, 3
        %C = xor i7 %B, 12
        store i7 %C, i7* %P
        %r = and i7 %C, 3
        ret i7 %r
}

define i7 @test6(i7 %A, i7 %B) {
        ;; ~(~X & Y) --> (X | ~Y)
        %t0 = xor i7 %A, -1
        %t1 = and i7 %t0, %B
        %r = xor i7 %t1, -1
        ret i7 %r
}

define i47 @test7(i47 %A) {
        %X = ashr i47 %A, 39 ;; sign extend
        %C1 = and i47 %X, 255
        ret i47 %C1
}
