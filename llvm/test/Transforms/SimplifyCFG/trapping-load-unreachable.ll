; RUN: llvm-as < %s | opt -simplifycfg | llvm-dis | grep {volatile load}
; PR2967

target datalayout =
"e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32"
target triple = "i386-pc-linux-gnu"

define void @foo(i32 %x) nounwind {
entry:
        %0 = icmp eq i32 %x, 0          ; <i1> [#uses=1]
        br i1 %0, label %bb, label %return

bb:             ; preds = %entry
        %1 = volatile load i32* null            ; <i32> [#uses=0]
        unreachable
        br label %return
return:         ; preds = %entry
        ret void
}
