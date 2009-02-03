; RUN: llvm-as < %s | opt -dse | llvm-dis | \
; RUN:    grep {store i32 1234567}

; Do not delete stores that are only partially killed.

define i32 @test() {
        %V = alloca i32         ; <i32*> [#uses=3]
        store i32 1234567, i32* %V
        %V2 = bitcast i32* %V to i8*            ; <i8*> [#uses=1]
        store i8 0, i8* %V2
        %X = load i32* %V               ; <i32> [#uses=1]
        ret i32 %X
}
