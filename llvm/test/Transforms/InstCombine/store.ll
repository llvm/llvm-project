; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:   grep -v {store.*,.*null} | not grep store

define void @test1(i32* %P) {
        store i32 undef, i32* %P
        store i32 123, i32* undef
        store i32 124, i32* null
        ret void
}

define void @test2(i32* %P) {
        %X = load i32* %P               ; <i32> [#uses=1]
        %Y = add i32 %X, 0              ; <i32> [#uses=1]
        store i32 %Y, i32* %P
        ret void
}

