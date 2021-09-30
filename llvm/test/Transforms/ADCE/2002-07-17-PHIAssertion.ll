; This testcase was extracted from the gzip SPEC benchmark
;
; RUN: opt < %s -passes=adce | FileCheck %s

@bk = external global i32               ; <i32*> [#uses=2]
@hufts = external global i32            ; <i32*> [#uses=1]

define i32 @inflate() {
bb0:
        br label %bb2

bb2:            ; preds = %bb6, %bb0
        %reg128 = phi i32 [ %reg130, %bb6 ], [ 0, %bb0 ]                ; <i32> [#uses=2]
        br i1 true, label %bb4, label %bb3

bb3:            ; preds = %bb2
        br label %UnifiedExitNode

; CHECK-NOT: bb4:
; CHECK-NOT: bb5:
bb4:            ; preds = %bb2
        %reg117 = load i32, i32* @hufts              ; <i32> [#uses=2]
        %cond241 = icmp ule i32 %reg117, %reg128                ; <i1> [#uses=1]
        br i1 %cond241, label %bb6, label %bb5

bb5:            ; preds = %bb4
        br label %bb6

bb6:            ; preds = %bb5, %bb4
        %reg130 = phi i32 [ %reg117, %bb5 ], [ %reg128, %bb4 ]          ; <i32> [#uses=1]
        br i1 false, label %bb2, label %bb7

bb7:            ; preds = %bb6
        %reg126 = load i32, i32* @bk         ; <i32> [#uses=1]
        %cond247 = icmp ule i32 %reg126, 7              ; <i1> [#uses=1]
        br i1 %cond247, label %bb9, label %bb8

bb8:            ; preds = %bb8, %bb7
        %reg119 = load i32, i32* @bk         ; <i32> [#uses=1]
        %cond256 = icmp ugt i32 %reg119, 7              ; <i1> [#uses=1]
        br i1 %cond256, label %bb8, label %bb9

bb9:            ; preds = %bb8, %bb7
        br label %UnifiedExitNode

UnifiedExitNode:                ; preds = %bb9, %bb3
        %UnifiedRetVal = phi i32 [ 7, %bb3 ], [ 0, %bb9 ]               ; <i32> [#uses=1]
        ret i32 %UnifiedRetVal
}

