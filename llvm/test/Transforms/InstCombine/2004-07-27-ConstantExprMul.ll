; RUN: opt < %s -instcombine -disable-output

@p = weak global i32 0          ; <i32*> [#uses=1]

define i32 @test(i32 %x) {
        %y = mul i32 %x, ptrtoint (i32* @p to i32)              ; <i32> [#uses=1]
        ret i32 %y
}

