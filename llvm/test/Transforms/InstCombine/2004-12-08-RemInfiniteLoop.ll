; RUN: opt < %s -passes=instcombine

define i32 @test(i32 %X) {
        %Y = srem i32 %X, undef         ; <i32> [#uses=1]
        ret i32 %Y
}

