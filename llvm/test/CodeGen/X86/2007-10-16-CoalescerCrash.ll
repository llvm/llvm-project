; RUN: llvm-as < %s | llc -mtriple=x86_64-apple-darwin

define i64 @__ashldi3(i64 %u, i64 %b) {
entry:
        br i1 false, label %UnifiedReturnBlock, label %cond_next

cond_next:              ; preds = %entry
        %tmp9 = sub i64 32, %b          ; <i64> [#uses=2]
        %tmp11 = icmp slt i64 %tmp9, 1          ; <i1> [#uses=1]
        %tmp2180 = trunc i64 %u to i32          ; <i32> [#uses=2]
        %tmp2223 = trunc i64 %tmp9 to i32               ; <i32> [#uses=2]
        br i1 %tmp11, label %cond_true14, label %cond_false

cond_true14:            ; preds = %cond_next
        %tmp24 = sub i32 0, %tmp2223            ; <i32> [#uses=1]
        %tmp25 = shl i32 %tmp2180, %tmp24               ; <i32> [#uses=1]
        %tmp2569 = zext i32 %tmp25 to i64               ; <i64> [#uses=1]
        %tmp256970 = shl i64 %tmp2569, 32               ; <i64> [#uses=1]
        ret i64 %tmp256970

cond_false:             ; preds = %cond_next
        %tmp35 = lshr i32 %tmp2180, %tmp2223            ; <i32> [#uses=1]
        %tmp54 = or i32 %tmp35, 0               ; <i32> [#uses=1]
        %tmp5464 = zext i32 %tmp54 to i64               ; <i64> [#uses=1]
        %tmp546465 = shl i64 %tmp5464, 32               ; <i64> [#uses=1]
        %tmp546465.ins = or i64 %tmp546465, 0           ; <i64> [#uses=1]
        ret i64 %tmp546465.ins

UnifiedReturnBlock:
        ret i64 %u
}
