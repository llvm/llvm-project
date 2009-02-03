; Test that elimination of logical operators works with 
; arbitrary precision integers.
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:    not grep {(and\|xor\|add\|shl\|shr)}
; END.

define i33 @test1(i33 %x) {
        %tmp.1 = and i33 %x, 65535           ; <i33> [#uses=1]
        %tmp.2 = xor i33 %tmp.1, -32768      ; <i33> [#uses=1]
        %tmp.3 = add i33 %tmp.2, 32768       ; <i33> [#uses=1]
        ret i33 %tmp.3
}

define i33 @test2(i33 %x) {
        %tmp.1 = and i33 %x, 65535           ; <i33> [#uses=1]
        %tmp.2 = xor i33 %tmp.1, 32768       ; <i33> [#uses=1]
        %tmp.3 = add i33 %tmp.2, -32768      ; <i33> [#uses=1]
        ret i33 %tmp.3
}

define i33 @test3(i16 %P) {
        %tmp.1 = zext i16 %P to i33          ; <i33> [#uses=1]
        %tmp.4 = xor i33 %tmp.1, 32768       ; <i33> [#uses=1]
        %tmp.5 = add i33 %tmp.4, -32768      ; <i33> [#uses=1]
        ret i33 %tmp.5
}

define i33 @test5(i33 %x) {
	%tmp.1 = and i33 %x, 254
	%tmp.2 = xor i33 %tmp.1, 128
	%tmp.3 = add i33 %tmp.2, -128
	ret i33 %tmp.3
}

define i33 @test6(i33 %x) {
        %tmp.2 = shl i33 %x, 16           ; <i33> [#uses=1]
        %tmp.4 = lshr i33 %tmp.2, 16      ; <i33> [#uses=1]
        ret i33 %tmp.4
}
