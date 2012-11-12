; RUN: llc < %s -march=ppc32 | not grep slwi

define i32 @test1(i64 %a) {
        %tmp29 = lshr i64 %a, 24                ; <i64> [#uses=1]
        %tmp23 = trunc i64 %tmp29 to i32                ; <i32> [#uses=1]
        %tmp410 = lshr i32 %tmp23, 9            ; <i32> [#uses=1]
        %tmp45 = trunc i32 %tmp410 to i16               ; <i16> [#uses=1]
        %tmp456 = sext i16 %tmp45 to i32                ; <i32> [#uses=1]
        ret i32 %tmp456
}

