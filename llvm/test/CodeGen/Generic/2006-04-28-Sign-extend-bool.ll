; RUN: llc < %s

define i32 @test(i32 %tmp93) {
        %tmp98 = shl i32 %tmp93, 31             ; <i32> [#uses=1]
        %tmp99 = ashr i32 %tmp98, 31            ; <i32> [#uses=1]
        %tmp99.upgrd.1 = trunc i32 %tmp99 to i8         ; <i8> [#uses=1]
        %tmp99100 = sext i8 %tmp99.upgrd.1 to i32               ; <i32> [#uses=1]
        ret i32 %tmp99100
}
