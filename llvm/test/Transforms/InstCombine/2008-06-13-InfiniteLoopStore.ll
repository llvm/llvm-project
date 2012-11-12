; RUN: opt < %s -instcombine -S | grep "store i32" | count 2

@g_139 = global i32 0           ; <i32*> [#uses=2]

define void @func_56(i32 %p_60) nounwind  {
entry:
        store i32 1, i32* @g_139, align 4
        %tmp1 = icmp ne i32 %p_60, 0            ; <i1> [#uses=1]
        %tmp12 = zext i1 %tmp1 to i8            ; <i8> [#uses=1]
        %toBool = icmp ne i8 %tmp12, 0          ; <i1> [#uses=1]
        br i1 %toBool, label %bb, label %return

bb:             ; preds = %bb, %entry
        store i32 1, i32* @g_139, align 4
        br label %bb

return:         ; preds = %entry
        ret void
}

