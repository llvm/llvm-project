; This test makes sure that add instructions are properly eliminated.
; This test is for Integer BitWidth > 64 && BitWidth <= 1024.

; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:    grep -v OK | not grep add
; END.

define i111 @test1(i111 %x) {
        %tmp.2 = shl i111 1, 110
        %tmp.4 = xor i111 %x, %tmp.2
        ;; Add of sign bit -> xor of sign bit.
        %tmp.6 = add i111 %tmp.4, %tmp.2
        ret i111 %tmp.6
}

define i65 @test2(i65 %x) {
        %tmp.0 = shl i65 1, 64
        %tmp.2 = xor i65 %x, %tmp.0
        ;; Add of sign bit -> xor of sign bit.
        %tmp.4 = add i65 %tmp.2, %tmp.0
        ret i65 %tmp.4
}

define i1024 @test3(i1024 %x) {
        %tmp.0 = shl i1024 1, 1023
        %tmp.2 = xor i1024 %x, %tmp.0
        ;; Add of sign bit -> xor of sign bit.
        %tmp.4 = add i1024 %tmp.2, %tmp.0
        ret i1024 %tmp.4
}

define i128 @test4(i128 %x) {
        ;; If we have ADD(XOR(AND(X, 0xFF), 0xF..F80), 0x80), it's a sext.
        %tmp.5 = shl i128 1, 127
        %tmp.1 = ashr i128 %tmp.5, 120
        %tmp.2 = xor i128 %x, %tmp.1      
        %tmp.4 = add i128 %tmp.2, %tmp.5
        ret i128 %tmp.4
}

define i77 @test6(i77 %x) {
        ;; (x & 254)+1 -> (x & 254)|1
        %tmp.2 = and i77 %x, 562949953421310
        %tmp.4 = add i77 %tmp.2, 1
        ret i77 %tmp.4
}
