; RUN: opt < %s -passes=instcombine -S | grep and
; PR913

define i32 @test(i32* %tmp1) {
        %tmp.i = load i32, i32* %tmp1                ; <i32> [#uses=1]
        %tmp = bitcast i32 %tmp.i to i32                ; <i32> [#uses=1]
        %tmp2.ui = lshr i32 %tmp, 5             ; <i32> [#uses=1]
        %tmp2 = bitcast i32 %tmp2.ui to i32             ; <i32> [#uses=1]
        %tmp3 = and i32 %tmp2, 1                ; <i32> [#uses=1]
        %tmp3.upgrd.1 = icmp ne i32 %tmp3, 0            ; <i1> [#uses=1]
        %tmp34 = zext i1 %tmp3.upgrd.1 to i32           ; <i32> [#uses=1]
        ret i32 %tmp34
}

