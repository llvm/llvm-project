; RUN: llvm-as < %s | opt -constprop | llvm-dis | \
; RUN:    grep -F {ret i32* null} | count 2

define i32* @test1() {
        %X = inttoptr i64 0 to i32*             ; <i32*> [#uses=1]
        ret i32* %X
}

define i32* @test2() {
        ret i32* null
}

