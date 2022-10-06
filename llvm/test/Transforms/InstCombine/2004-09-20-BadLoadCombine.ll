; RUN: opt < %s -instcombine -mem2reg -S | \
; RUN:   not grep "i32 1"

; When propagating the load through the select, make sure that the load is
; inserted where the original load was, not where the select is.  Not doing
; so could produce incorrect results!

define i32 @test(i1 %C) {
        %X = alloca i32         ; <ptr> [#uses=3]
        %X2 = alloca i32                ; <ptr> [#uses=2]
        store i32 1, ptr %X
        store i32 2, ptr %X2
        %Y = select i1 %C, ptr %X, ptr %X2            ; <ptr> [#uses=1]
        store i32 3, ptr %X
        %Z = load i32, ptr %Y               ; <i32> [#uses=1]
        ret i32 %Z
}

