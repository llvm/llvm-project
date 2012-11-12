; RUN: llc < %s -march=arm | \
; RUN:   grep "str.*\!" | count 2

define void @test1(i32* %X, i32* %A, i32** %dest) {
        %B = load i32* %A               ; <i32> [#uses=1]
        %Y = getelementptr i32* %X, i32 4               ; <i32*> [#uses=2]
        store i32 %B, i32* %Y
        store i32* %Y, i32** %dest
        ret void
}

define i16* @test2(i16* %X, i32* %A) {
        %B = load i32* %A               ; <i32> [#uses=1]
        %Y = getelementptr i16* %X, i32 4               ; <i16*> [#uses=2]
        %tmp = trunc i32 %B to i16              ; <i16> [#uses=1]
        store i16 %tmp, i16* %Y
        ret i16* %Y
}
