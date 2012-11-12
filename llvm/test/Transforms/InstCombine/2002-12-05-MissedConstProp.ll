; RUN: opt < %s -instcombine -S | not grep add

define i32 @test(i32 %A) {
        %A.neg = sub i32 0, %A          ; <i32> [#uses=1]
        %.neg = sub i32 0, 1            ; <i32> [#uses=1]
        %X = add i32 %.neg, 1           ; <i32> [#uses=1]
        %Y.neg.ra = add i32 %A, %X              ; <i32> [#uses=1]
        %r = add i32 %A.neg, %Y.neg.ra          ; <i32> [#uses=1]
        ret i32 %r
}

