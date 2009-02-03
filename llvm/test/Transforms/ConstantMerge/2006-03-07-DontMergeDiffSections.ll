; RUN: llvm-as < %s | opt -constmerge | llvm-dis | grep foo
; RUN: llvm-as < %s | opt -constmerge | llvm-dis | grep bar

; Don't merge constants in different sections.

@G1 = internal constant i32 1, section "foo"            ; <i32*> [#uses=1]
@G2 = internal constant i32 1, section "bar"            ; <i32*> [#uses=1]
@G3 = internal constant i32 1, section "bar"            ; <i32*> [#uses=1]

define void @test(i32** %P1, i32** %P2, i32** %P3) {
        store i32* @G1, i32** %P1
        store i32* @G2, i32** %P2
        store i32* @G3, i32** %P3
        ret void
}

