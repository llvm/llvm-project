; Test that pure functions are cse'd away
; RUN: llvm-as < %s | opt -globalsmodref-aa -gvn -instcombine | \
; RUN: llvm-dis | not grep sub

define i32 @pure(i32 %X) {
        %Y = add i32 %X, 1              ; <i32> [#uses=1]
        ret i32 %Y
}

define i32 @test1(i32 %X) {
        %A = call i32 @pure( i32 %X )           ; <i32> [#uses=1]
        %B = call i32 @pure( i32 %X )           ; <i32> [#uses=1]
        %C = sub i32 %A, %B             ; <i32> [#uses=1]
        ret i32 %C
}

define i32 @test2(i32 %X, i32* %P) {
        %A = call i32 @pure( i32 %X )           ; <i32> [#uses=1]
        store i32 %X, i32* %P ;; Does not invalidate 'pure' call.
        %B = call i32 @pure( i32 %X )           ; <i32> [#uses=1]
        %C = sub i32 %A, %B             ; <i32> [#uses=1]
        ret i32 %C
}
