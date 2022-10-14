; RUN: opt < %s -passes=indvars -S | FileCheck %s

define i32 @main() {
; CHECK-LABEL: @main(
; CHECK: ret i32 152
entry:
        br label %no_exit

no_exit:                ; preds = %no_exit, %entry
        %i.1.0 = phi i32 [ 0, %entry ], [ %inc, %no_exit ]              ; <i32> [#uses=2]
        %tmp.4 = icmp sgt i32 %i.1.0, 50                ; <i1> [#uses=1]
        %tmp.7 = select i1 %tmp.4, i32 100, i32 0               ; <i32> [#uses=1]
        %i.0 = add i32 %i.1.0, 1                ; <i32> [#uses=1]
        %inc = add i32 %i.0, %tmp.7             ; <i32> [#uses=3]
        %tmp.1 = icmp slt i32 %inc, 100         ; <i1> [#uses=1]
        br i1 %tmp.1, label %no_exit, label %loopexit

loopexit:               ; preds = %no_exit
        ret i32 %inc
}

