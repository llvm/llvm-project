; RUN: llvm-as < %s | llc -march=c | not grep -- --65535
; PR596

target datalayout = "e-p:32:32"
target triple = "i686-pc-linux-gnu"

declare void @func(i32)

define void @funcb() {
entry:
        %tmp.1 = sub i32 0, -65535              ; <i32> [#uses=1]
        call void @func( i32 %tmp.1 )
        br label %return

return:         ; preds = %entry
        ret void
}

