; PR726
; RUN: opt < %s -passes=indvars -S | FileCheck %s

; Make sure to compute the right exit value based on negative strides.

define i32 @test() {
; CHECK-LABEL: @test(
; CHECK: ret i32 27
entry:
        br label %cond_true

cond_true:              ; preds = %cond_true, %entry
        %a.0.0 = phi i32 [ 10, %entry ], [ %tmp4, %cond_true ]          ; <i32> [#uses=2]
        %b.0.0 = phi i32 [ 0, %entry ], [ %tmp2, %cond_true ]           ; <i32> [#uses=1]
        %tmp2 = add i32 %b.0.0, %a.0.0          ; <i32> [#uses=2]
        %tmp4 = add i32 %a.0.0, -1              ; <i32> [#uses=2]
        %tmp = icmp sgt i32 %tmp4, 7            ; <i1> [#uses=1]
        br i1 %tmp, label %cond_true, label %bb7

bb7:            ; preds = %cond_true
        ret i32 %tmp2
}

