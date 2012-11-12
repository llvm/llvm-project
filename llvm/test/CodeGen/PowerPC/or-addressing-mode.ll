; RUN: llc < %s -mtriple=powerpc-apple-darwin8 | not grep ori
; RUN: llc < %s -mtriple=powerpc-apple-darwin8 | not grep rlwimi

define i32 @test1(i8* %P) {
        %tmp.2.i = ptrtoint i8* %P to i32               ; <i32> [#uses=2]
        %tmp.4.i = and i32 %tmp.2.i, -65536             ; <i32> [#uses=1]
        %tmp.10.i = lshr i32 %tmp.2.i, 5                ; <i32> [#uses=1]
        %tmp.11.i = and i32 %tmp.10.i, 2040             ; <i32> [#uses=1]
        %tmp.13.i = or i32 %tmp.11.i, %tmp.4.i          ; <i32> [#uses=1]
        %tmp.14.i = inttoptr i32 %tmp.13.i to i32*              ; <i32*> [#uses=1]
        %tmp.3 = load i32* %tmp.14.i            ; <i32> [#uses=1]
        ret i32 %tmp.3
}

define i32 @test2(i32 %P) {
        %tmp.2 = shl i32 %P, 4          ; <i32> [#uses=1]
        %tmp.3 = or i32 %tmp.2, 2               ; <i32> [#uses=1]
        %tmp.4 = inttoptr i32 %tmp.3 to i32*            ; <i32*> [#uses=1]
        %tmp.5 = load i32* %tmp.4               ; <i32> [#uses=1]
        ret i32 %tmp.5
}

