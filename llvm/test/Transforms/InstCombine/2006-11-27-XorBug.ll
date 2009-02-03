; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep and.*32
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:    not grep or.*153
; PR1014

define i32 @test(i32 %tmp1) {
        %ovm = and i32 %tmp1, 32                ; <i32> [#uses=1]
        %ov3 = add i32 %ovm, 145                ; <i32> [#uses=1]
        %ov110 = xor i32 %ov3, 153              ; <i32> [#uses=1]
        ret i32 %ov110
}

