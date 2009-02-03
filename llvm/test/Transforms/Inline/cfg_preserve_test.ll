; This test ensures that inlining an "empty" function does not destroy the CFG
;
; RUN: llvm-as < %s | opt -inline | llvm-dis | not grep br

define i32 @func(i32 %i) {
        ret i32 %i
}

declare void @bar()

define i32 @main(i32 %argc) {
Entry:
        %X = call i32 @func( i32 7 )            ; <i32> [#uses=1]
        ret i32 %X
}

