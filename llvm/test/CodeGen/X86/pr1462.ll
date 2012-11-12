; RUN: llc < %s
; PR1462

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-
v64:64:64-v128:128:128-a0:0:64"
target triple = "x86_64-unknown-linux-gnu"

define hidden i128 @__addvti3(i128 %a1, i128 %b2) {
entry:
        %tmp8 = add i128 %b2, %a1               ; <i128> [#uses=3]
        %tmp10 = icmp sgt i128 %b2, -1          ; <i1> [#uses=1]
        %tmp18 = icmp sgt i128 %tmp8, %a1               ; <i1> [#uses=1]
        %tmp14 = icmp slt i128 %tmp8, %a1               ; <i1> [#uses=1]
        %iftmp.0.0.in = select i1 %tmp10, i1 %tmp14, i1 %tmp18          ; <i1> [#uses=1]
        br i1 %iftmp.0.0.in, label %cond_true22, label %cond_next23

cond_true22:            ; preds = %entry
        tail call void @abort( )
        unreachable

cond_next23:            ; preds = %entry
        ret i128 %tmp8
}

declare void @abort()
