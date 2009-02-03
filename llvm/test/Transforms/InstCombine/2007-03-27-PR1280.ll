; PR1280 - we should be able to reduce this function to a trunc/sext but it
;          would involve using a bit width (24) that doesn't match a size that
;          the back end can handle. This test makes sure that such a transform
;          is not done. It should be removed when code gen supports "funny"
;          bit widths.

; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {add i49.*-8388608}

define i49 @test5(i49 %x) {
        ;; If we have ADD(XOR(AND(X, 0xFF), 0x80), 0xF..F80), it's a sext.
        %X = and i49 %x, 16777215                     ; 0x0000000ffffff
        %tmp.2 = xor i49 %X, 8388608                  ; 0x0000000800000
        %tmp.4 = add i49 %tmp.2, -8388608             ; 0x1FFFFFF800000
        ret i49 %tmp.4
}
