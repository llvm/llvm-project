; RUN: llvm-as < %s | opt -inline | llvm-dis | grep call

; 'bar' can be overridden at link-time, don't inline it.

define void @foo() {
entry:
        tail call void @bar( )            ; <i32> [#uses=0]
        ret void
}

define weak void @bar() {
        ret void
}

