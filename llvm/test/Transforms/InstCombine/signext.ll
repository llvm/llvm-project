; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:    not grep {(and\|xor\|add\|shl\|shr)}
; END.

define i32 @test1(i32 %x) {
        %tmp.1 = and i32 %x, 65535              ; <i32> [#uses=1]
        %tmp.2 = xor i32 %tmp.1, -32768         ; <i32> [#uses=1]
        %tmp.3 = add i32 %tmp.2, 32768          ; <i32> [#uses=1]
        ret i32 %tmp.3
}

define i32 @test2(i32 %x) {
        %tmp.1 = and i32 %x, 65535              ; <i32> [#uses=1]
        %tmp.2 = xor i32 %tmp.1, 32768          ; <i32> [#uses=1]
        %tmp.3 = add i32 %tmp.2, -32768         ; <i32> [#uses=1]
        ret i32 %tmp.3
}

define i32 @test3(i16 %P) {
        %tmp.1 = zext i16 %P to i32             ; <i32> [#uses=1]
        %tmp.4 = xor i32 %tmp.1, 32768          ; <i32> [#uses=1]
        %tmp.5 = add i32 %tmp.4, -32768         ; <i32> [#uses=1]
        ret i32 %tmp.5
}

define i32 @test4(i16 %P) {
        %tmp.1 = zext i16 %P to i32             ; <i32> [#uses=1]
        %tmp.4 = xor i32 %tmp.1, 32768          ; <i32> [#uses=1]
        %tmp.5 = add i32 %tmp.4, -32768         ; <i32> [#uses=1]
        ret i32 %tmp.5
}

define i32 @test5(i32 %x) {
        %tmp.1 = and i32 %x, 254                ; <i32> [#uses=1]
        %tmp.2 = xor i32 %tmp.1, 128            ; <i32> [#uses=1]
        %tmp.3 = add i32 %tmp.2, -128           ; <i32> [#uses=1]
        ret i32 %tmp.3
}

define i32 @test6(i32 %x) {
        %tmp.2 = shl i32 %x, 16         ; <i32> [#uses=1]
        %tmp.4 = ashr i32 %tmp.2, 16            ; <i32> [#uses=1]
        ret i32 %tmp.4
}

