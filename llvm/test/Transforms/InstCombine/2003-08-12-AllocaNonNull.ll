; This testcase can be simplified by "realizing" that alloca can never return
; null.
; RUN: opt < %s -instcombine -simplifycfg -S | FileCheck %s
; CHECK-NOT: br

declare i32 @bitmap_clear(...)

define i32 @oof() {
entry:
        %live_head = alloca i32         ; <i32*> [#uses=2]
        %tmp.1 = icmp ne i32* %live_head, null          ; <i1> [#uses=1]
        br i1 %tmp.1, label %then, label %UnifiedExitNode

then:           ; preds = %entry
        %tmp.4 = call i32 (...) @bitmap_clear( i32* %live_head )               ; <i32> [#uses=0]
        br label %UnifiedExitNode

UnifiedExitNode:                ; preds = %then, %entry
        ret i32 0
}

