; RUN: opt < %s -instcombine -S | \
; RUN:   not grep undef

define i32 @test(i8 %A) {
        %B = sext i8 %A to i32          ; <i32> [#uses=1]
        %C = ashr i32 %B, 8             ; <i32> [#uses=1]
        ret i32 %C
}


