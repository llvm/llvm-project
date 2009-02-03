; RUN: llvm-as < %s | llc -march=arm -mattr=+v6 | grep rev16
; RUN: llvm-as < %s | llc -march=arm -mattr=+v6 | grep revsh

define i32 @test1(i32 %X) {
        %tmp1 = lshr i32 %X, 8          ; <i32> [#uses=3]
        %X15 = bitcast i32 %X to i32            ; <i32> [#uses=1]
        %tmp4 = shl i32 %X15, 8         ; <i32> [#uses=2]
        %tmp2 = and i32 %tmp1, 16711680         ; <i32> [#uses=1]
        %tmp5 = and i32 %tmp4, -16777216                ; <i32> [#uses=1]
        %tmp9 = and i32 %tmp1, 255              ; <i32> [#uses=1]
        %tmp13 = and i32 %tmp4, 65280           ; <i32> [#uses=1]
        %tmp6 = or i32 %tmp5, %tmp2             ; <i32> [#uses=1]
        %tmp10 = or i32 %tmp6, %tmp13           ; <i32> [#uses=1]
        %tmp14 = or i32 %tmp10, %tmp9           ; <i32> [#uses=1]
        ret i32 %tmp14
}

define i32 @test2(i32 %X) {
        %tmp1 = lshr i32 %X, 8          ; <i32> [#uses=1]
        %tmp1.upgrd.1 = trunc i32 %tmp1 to i16          ; <i16> [#uses=1]
        %tmp3 = trunc i32 %X to i16             ; <i16> [#uses=1]
        %tmp2 = and i16 %tmp1.upgrd.1, 255              ; <i16> [#uses=1]
        %tmp4 = shl i16 %tmp3, 8                ; <i16> [#uses=1]
        %tmp5 = or i16 %tmp2, %tmp4             ; <i16> [#uses=1]
        %tmp5.upgrd.2 = sext i16 %tmp5 to i32           ; <i32> [#uses=1]
        ret i32 %tmp5.upgrd.2
}
