; RUN: llc -verify-machineinstrs < %s | grep xor

target datalayout = "E-p:32:32"
target triple = "powerpc-apple-darwin8.7.0"

define void @foo(i32 %X) {
entry:
        %tmp1 = and i32 %X, 3           ; <i32> [#uses=1]
        %tmp2 = xor i32 %tmp1, 1                ; <i32> [#uses=1]
        %tmp = icmp eq i32 %tmp2, 0             ; <i1> [#uses=1]
        br i1 %tmp, label %UnifiedReturnBlock, label %cond_true
cond_true:              ; preds = %entry
        tail call i32 (...) @bar( )            ; <i32>:0 [#uses=0]
        ret void
UnifiedReturnBlock:             ; preds = %entry
        ret void
}

declare i32 @bar(...)

