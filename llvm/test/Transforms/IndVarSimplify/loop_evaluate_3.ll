; RUN: opt < %s -indvars -S | grep "ret i32 600000"
; PR1179

define i32 @foo() {
entry:
        br label %bb5

bb5:            ; preds = %bb5, %entry
        %i.01.0 = phi i32 [ 0, %entry ], [ %tmp2, %bb5 ]                ; <i32> [#uses=1]
        %x.03.0 = phi i32 [ 0, %entry ], [ %tmp4, %bb5 ]                ; <i32> [#uses=1]
        %tmp2 = add i32 %i.01.0, 3              ; <i32> [#uses=2]
        %tmp4 = add i32 %x.03.0, 1              ; <i32> [#uses=2]
        icmp slt i32 %tmp4, 200000              ; <i1>:0 [#uses=1]
        br i1 %0, label %bb5, label %bb7

bb7:            ; preds = %bb5
        ret i32 %tmp2
}

