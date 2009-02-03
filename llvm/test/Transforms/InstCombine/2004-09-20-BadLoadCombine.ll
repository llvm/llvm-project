; RUN: llvm-as < %s | opt -instcombine -mem2reg | llvm-dis | \
; RUN:   not grep {i32 1}

; When propagating the load through the select, make sure that the load is
; inserted where the original load was, not where the select is.  Not doing
; so could produce incorrect results!

define i32 @test(i1 %C) {
        %X = alloca i32         ; <i32*> [#uses=3]
        %X2 = alloca i32                ; <i32*> [#uses=2]
        store i32 1, i32* %X
        store i32 2, i32* %X2
        %Y = select i1 %C, i32* %X, i32* %X2            ; <i32*> [#uses=1]
        store i32 3, i32* %X
        %Z = load i32* %Y               ; <i32> [#uses=1]
        ret i32 %Z
}

