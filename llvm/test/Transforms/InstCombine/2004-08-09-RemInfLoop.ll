; RUN: opt < %s -passes=instcombine

; This testcase should not send the instcombiner into an infinite loop!

define i32 @test(i32 %X) {
        %Y = srem i32 %X, 0             ; <i32> [#uses=1]
        ret i32 %Y
}

