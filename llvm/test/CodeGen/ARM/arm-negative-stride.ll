; RUN: llc -mtriple=arm-eabi %s -o - | FileCheck %s

; This loop is rewritten with an indvar which counts down, which
; frees up a register from holding the trip count.

define void @test(ptr %P, i32 %A, i32 %i) nounwind {
entry:
; CHECK: str r1, [{{r.*}}, {{r.*}}, lsl #2]
        icmp eq i32 %i, 0               ; <i1>:0 [#uses=1]
        br i1 %0, label %return, label %bb

bb:             ; preds = %bb, %entry
        %indvar = phi i32 [ 0, %entry ], [ %indvar.next, %bb ]          ; <i32> [#uses=2]
        %i_addr.09.0 = sub i32 %i, %indvar              ; <i32> [#uses=1]
        %tmp2 = getelementptr i32, ptr %P, i32 %i_addr.09.0         ; <ptr> [#uses=1]
        store i32 %A, ptr %tmp2
        %indvar.next = add i32 %indvar, 1               ; <i32> [#uses=2]
        icmp eq i32 %indvar.next, %i            ; <i1>:1 [#uses=1]
        br i1 %1, label %return, label %bb

return:         ; preds = %bb, %entry
        ret void
}

; This loop has a non-address use of the count-up indvar, so
; it'll remain. Now the original store uses a negative-stride address.

define void @test_with_forced_iv(ptr %P, i32 %A, i32 %i) nounwind {
entry:
; CHECK: str r1, [{{r.*}}, -{{r.*}}, lsl #2]
        icmp eq i32 %i, 0               ; <i1>:0 [#uses=1]
        br i1 %0, label %return, label %bb

bb:             ; preds = %bb, %entry
        %indvar = phi i32 [ 0, %entry ], [ %indvar.next, %bb ]          ; <i32> [#uses=2]
        %i_addr.09.0 = sub i32 %i, %indvar              ; <i32> [#uses=1]
        %tmp2 = getelementptr i32, ptr %P, i32 %i_addr.09.0         ; <ptr> [#uses=1]
        store i32 %A, ptr %tmp2
        store i32 %indvar, ptr null
        %indvar.next = add i32 %indvar, 1               ; <i32> [#uses=2]
        icmp eq i32 %indvar.next, %i            ; <i1>:1 [#uses=1]
        br i1 %1, label %return, label %bb

return:         ; preds = %bb, %entry
        ret void
}

