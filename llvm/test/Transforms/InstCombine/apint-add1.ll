; This test makes sure that add instructions are properly eliminated.
; This test is for Integer BitWidth <= 64 && BitWidth % 8 != 0.

; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:    grep -v OK | not grep add


define i1 @test1(i1 %x) {
        %tmp.2 = xor i1 %x, 1
        ;; Add of sign bit -> xor of sign bit.
        %tmp.4 = add i1 %tmp.2, 1
        ret i1 %tmp.4
}

define i47 @test2(i47 %x) {
        %tmp.2 = xor i47 %x, 70368744177664
        ;; Add of sign bit -> xor of sign bit.
        %tmp.4 = add i47 %tmp.2, 70368744177664
        ret i47 %tmp.4
}

define i15 @test3(i15 %x) {
        %tmp.2 = xor i15 %x, 16384
        ;; Add of sign bit -> xor of sign bit.
        %tmp.4 = add i15 %tmp.2, 16384
        ret i15 %tmp.4
}

define i49 @test6(i49 %x) {
        ;; (x & 254)+1 -> (x & 254)|1
        %tmp.2 = and i49 %x, 562949953421310
        %tmp.4 = add i49 %tmp.2, 1
        ret i49 %tmp.4
}
