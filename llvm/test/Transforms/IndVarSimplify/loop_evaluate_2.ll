; RUN: opt < %s -passes='loop(loop-deletion),simplifycfg' -simplifycfg-require-and-preserve-domtree=1 | opt -passes='print<loops>' -disable-output 2>&1 | FileCheck %s
; PR1179

; CHECK-NOT: Loop Containing

define i32 @ltst(i32 %x) {
entry:
        icmp sgt i32 %x, 0              ; <i1>:0 [#uses=1]
        br i1 %0, label %bb.preheader, label %bb8

bb.preheader:           ; preds = %entry
        br label %bb

bb:             ; preds = %bb, %bb.preheader
        %i.01.0 = phi i32 [ %tmp4, %bb ], [ 0, %bb.preheader ]          ; <i32> [#uses=1]
        %j.03.0 = phi i32 [ %tmp2, %bb ], [ 0, %bb.preheader ]          ; <i32> [#uses=1]
        %tmp4 = add i32 %i.01.0, 1              ; <i32> [#uses=2]
        %tmp2 = add i32 %j.03.0, 1              ; <i32> [#uses=2]
        icmp slt i32 %tmp4, %x          ; <i1>:1 [#uses=1]
        br i1 %1, label %bb, label %bb8.loopexit

bb8.loopexit:           ; preds = %bb
        br label %bb8

bb8:            ; preds = %bb8.loopexit, %entry
        %j.03.1 = phi i32 [ 0, %entry ], [ %tmp2, %bb8.loopexit ]               ; <i32> [#uses=1]
        ret i32 %j.03.1
}

