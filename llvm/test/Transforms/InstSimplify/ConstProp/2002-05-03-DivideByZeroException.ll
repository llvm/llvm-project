; Make sure that the constant propogator doesn't divide by zero!
;
; RUN: opt < %s -passes=instsimplify
;

define i32 @test() {
        %R = sdiv i32 12, 0             ; <i32> [#uses=1]
        ret i32 %R
}

define i32 @test2() {
        %R = srem i32 12, 0             ; <i32> [#uses=1]
        ret i32 %R
}

