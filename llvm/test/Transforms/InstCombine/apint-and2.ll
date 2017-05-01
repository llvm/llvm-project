; This test makes sure that and instructions are properly eliminated.
; This test is for Integer BitWidth > 64 && BitWidth <= 1024.

; RUN: opt < %s -instcombine -S | not grep "and "
; END.


define i999 @test0(i999 %A) {
        %B = and i999 %A, 0 ; zero result
        ret i999 %B
}

define i1005 @test2(i1005 %x) {
        %tmp.2 = and i1005 %x, -1 ; noop
        ret i1005 %tmp.2
}

define i123 @test3(i123 %x) {
        %tmp.0 = and i123 %x, 127
        %tmp.2 = and i123 %tmp.0, 128
        ret i123 %tmp.2
}

define i1 @test4(i737 %x) {
        %A = and i737 %x, -2147483648
        %B = icmp ne i737 %A, 0
        ret i1 %B
}

define i117 @test5(i117 %A, i117* %P) {
        %B = or i117 %A, 3
        %C = xor i117 %B, 12
        store i117 %C, i117* %P
        %r = and i117 %C, 3
        ret i117 %r
}

define i117 @test6(i117 %A, i117 %B) {
        ;; ~(~X & Y) --> (X | ~Y)
        %t0 = xor i117 %A, -1
        %t1 = and i117 %t0, %B
        %r = xor i117 %t1, -1
        ret i117 %r
}

define i1024 @test7(i1024 %A) {
        %X = ashr i1024 %A, 1016 ;; sign extend
        %C1 = and i1024 %X, 255
        ret i1024 %C1
}
