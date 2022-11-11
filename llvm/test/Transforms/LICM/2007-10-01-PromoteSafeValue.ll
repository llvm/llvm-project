; RUN: opt < %s -licm -S | FileCheck %s
; Promote value if at least one use is safe


define i32 @f2(ptr %p, ptr %q) {
entry:
        br label %loop.head

loop.head:              ; preds = %cond.true, %entry
        store i32 20, ptr %p
        %tmp3.i = icmp eq ptr null, %q            ; <i1> [#uses=1]
        br i1 %tmp3.i, label %exit, label %cond.true
        
cond.true:              ; preds = %loop.head
        store i32 40, ptr %p
        br label %loop.head

; CHECK: exit:
; CHECK: store i32 20, ptr %p
exit:           ; preds = %loop.head
        ret i32 0
}

