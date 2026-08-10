; RUN: opt < %s -passes=instcombine -S | not grep 34

define i32 @test(i32 %X) {
        ; Do not fold into shr X, 34, as this uses undefined behavior!
        %Y = ashr i32 %X, 17            ; <i32> [#uses=1]
        %Z = ashr i32 %Y, 17            ; <i32> [#uses=1]
        ret i32 %Z
}

define i32 @test2(i32 %X) {
        ; Do not fold into shl X, 34, as this uses undefined behavior!
        %Y = shl i32 %X, 17             ; <i32> [#uses=1]
        %Z = shl i32 %Y, 17             ; <i32> [#uses=1]
        ret i32 %Z
}
