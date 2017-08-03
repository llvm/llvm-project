; RUN: llc < %s -march=x86 | not grep leal

@x = external global i32                ; <i32*> [#uses=1]

define i32 @test() {
        %tmp.0 = load i32, i32* @x           ; <i32> [#uses=1]
        %tmp.1 = shl i32 %tmp.0, 1              ; <i32> [#uses=1]
        ret i32 %tmp.1
}

