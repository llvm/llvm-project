; RUN: opt < %s -instcombine -S | grep "zext " | count 1
; PR1570

define i32 @test2(float %X, float %Y) {
entry:
        %tmp3 = fcmp uno float %X, %Y           ; <i1> [#uses=1]
        %tmp34 = zext i1 %tmp3 to i8            ; <i8> [#uses=1]
        %tmp = xor i8 %tmp34, 1         ; <i8> [#uses=1]
        %toBoolnot5 = zext i8 %tmp to i32               ; <i32> [#uses=1]
        ret i32 %toBoolnot5
}

