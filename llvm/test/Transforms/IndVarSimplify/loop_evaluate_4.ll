; RUN: opt < %s -indvars -S | FileCheck %s
; PR1179

define i32 @test4() {
; CHECK-LABEL: @test4(
; CHECK: ret i32 9900
entry:
        br label %bb7

bb7:            ; preds = %bb7, %entry
        %v.01.0 = phi i32 [ 0, %entry ], [ %tmp4, %bb7 ]                ; <i32> [#uses=1]
        %i.03.0 = phi i32 [ 0, %entry ], [ %tmp6, %bb7 ]                ; <i32> [#uses=2]
        %tmp2 = shl i32 %i.03.0, 1              ; <i32> [#uses=1]
        %tmp4 = add i32 %tmp2, %v.01.0          ; <i32> [#uses=2]
        %tmp6 = add i32 %i.03.0, 1              ; <i32> [#uses=2]
        icmp slt i32 %tmp6, 100         ; <i1>:0 [#uses=1]
        br i1 %0, label %bb7, label %bb9

bb9:            ; preds = %bb7
        ret i32 %tmp4
}

