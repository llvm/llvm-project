; RUN: llvm-as < %s | opt -inline | llvm-dis | not grep callee
; RUN: llvm-as < %s | opt -inline | llvm-dis | not grep div


define internal i32 @callee(i32 %A, i32 %B) {
        %C = sdiv i32 %A, %B            ; <i32> [#uses=1]
        ret i32 %C
}

define i32 @test() {
        %X = call i32 @callee( i32 10, i32 3 )          ; <i32> [#uses=1]
        ret i32 %X
}

