; RUN: llc < %s -march=x86 -mcpu=i386 | \
; RUN:    not grep "movl %eax, %edx"

define i32 @foo(i32 %t, i32 %C) {
entry:
        br label %cond_true

cond_true:              ; preds = %cond_true, %entry
        %t_addr.0.0 = phi i32 [ %t, %entry ], [ %tmp7, %cond_true ]             ; <i32> [#uses=2]
        %tmp7 = add i32 %t_addr.0.0, 1          ; <i32> [#uses=1]
        %tmp = icmp sgt i32 %C, 39              ; <i1> [#uses=1]
        br i1 %tmp, label %bb12, label %cond_true

bb12:           ; preds = %cond_true
        ret i32 %t_addr.0.0
}

