; Test that functions with dynamic allocas get inlined in a case where
; naively inlining it would result in a miscompilation.
; Functions with dynamic allocas can only be inlined into functions that
; already have dynamic allocas.

; RUN: llvm-as < %s | opt -inline | llvm-dis | \
; RUN:   grep llvm.stacksave
; RUN: llvm-as < %s | opt -inline | llvm-dis | not grep callee


declare void @ext(i32*)

define internal void @callee(i32 %N) {
        %P = alloca i32, i32 %N         ; <i32*> [#uses=1]
        call void @ext( i32* %P )
        ret void
}

define void @foo(i32 %N) {
; <label>:0
        %P = alloca i32, i32 %N         ; <i32*> [#uses=1]
        call void @ext( i32* %P )
        br label %Loop

Loop:           ; preds = %Loop, %0
        %count = phi i32 [ 0, %0 ], [ %next, %Loop ]            ; <i32> [#uses=2]
        %next = add i32 %count, 1               ; <i32> [#uses=1]
        call void @callee( i32 %N )
        %cond = icmp eq i32 %count, 100000              ; <i1> [#uses=1]
        br i1 %cond, label %out, label %Loop

out:            ; preds = %Loop
        ret void
}

