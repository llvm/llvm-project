; Tests to make sure elimination of casts is working correctly
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | notcast

define i17 @test1(i17 %a) {
        %tmp = zext i17 %a to i37               ; <i37> [#uses=2]
        %tmp21 = lshr i37 %tmp, 8               ; <i37> [#uses=1]
        %tmp5 = shl i37 %tmp, 8         ; <i37> [#uses=1]
        %tmp.upgrd.32 = or i37 %tmp21, %tmp5            ; <i37> [#uses=1]
        %tmp.upgrd.3 = trunc i37 %tmp.upgrd.32 to i17   ; <i17> [#uses=1]
        ret i17 %tmp.upgrd.3
}

define i167 @test2(i167 %a) {
        %tmp = zext i167 %a to i577               ; <i577> [#uses=2]
        %tmp21 = lshr i577 %tmp, 9               ; <i577> [#uses=1]
        %tmp5 = shl i577 %tmp, 8         ; <i577> [#uses=1]
        %tmp.upgrd.32 = or i577 %tmp21, %tmp5            ; <i577> [#uses=1]
        %tmp.upgrd.3 = trunc i577 %tmp.upgrd.32 to i167  ; <i167> [#uses=1]
        ret i167 %tmp.upgrd.3
}
