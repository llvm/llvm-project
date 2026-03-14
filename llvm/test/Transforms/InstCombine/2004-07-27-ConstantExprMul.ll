; RUN: opt < %s -passes=instcombine -disable-output

@p = weak global i32 0          ; <ptr> [#uses=1]

define i32 @test(i32 %x) {
        %y = mul i32 %x, ptrtoint (ptr @p to i32)              ; <i32> [#uses=1]
        ret i32 %y
}

