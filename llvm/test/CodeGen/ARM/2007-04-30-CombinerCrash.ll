; RUN: llc < %s -mtriple=arm-apple-darwin -mattr=+v6,+vfp2

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:64"
target triple = "arm-apple-darwin8"
        %struct.CHESS_POSITION = type { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32, i32, i8, i8, [64 x i8], i8, i8, i8, i8, i8 }
@search = external global %struct.CHESS_POSITION                ; <ptr> [#uses=3]
@file_mask = external global [8 x i64]          ; <ptr> [#uses=1]
@rank_mask.1.b = external global i1             ; <ptr> [#uses=1]

define fastcc void @EvaluateDevelopment() {
entry:
        %tmp7 = load i64, ptr getelementptr (%struct.CHESS_POSITION, ptr @search, i32 0, i32 7)         ; <i64> [#uses=1]
        %tmp50 = load i64, ptr @search                ; <i64> [#uses=1]
        %tmp52 = load i64, ptr getelementptr (%struct.CHESS_POSITION, ptr @search, i32 0, i32 1)                ; <i64> [#uses=1]
        %tmp53 = or i64 %tmp52, %tmp50          ; <i64> [#uses=1]
        %tmp57.b = load i1, ptr @rank_mask.1.b              ; <i1> [#uses=1]
        %tmp57 = select i1 %tmp57.b, i64 71776119061217280, i64 0               ; <i64> [#uses=1]
        %tmp58 = and i64 %tmp57, %tmp7          ; <i64> [#uses=1]
        %tmp59 = lshr i64 %tmp58, 8             ; <i64> [#uses=1]
        %tmp63 = load i64, ptr getelementptr ([8 x i64], ptr @file_mask, i32 0, i32 4)          ; <i64> [#uses=1]
        %tmp64 = or i64 %tmp63, 0               ; <i64> [#uses=1]
        %tmp65 = and i64 %tmp59, %tmp53         ; <i64> [#uses=1]
        %tmp66 = and i64 %tmp65, %tmp64         ; <i64> [#uses=1]
        %tmp67 = icmp eq i64 %tmp66, 0          ; <i1> [#uses=1]
        br i1 %tmp67, label %cond_next145, label %cond_true70

cond_true70:            ; preds = %entry
        ret void

cond_next145:           ; preds = %entry
        ret void
}
