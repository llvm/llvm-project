; The i induction variable looks like a wrap-around, but it really is just
; a simple affine IV.  Make sure that indvars eliminates it.

; RUN: llvm-as < %s | opt -indvars | llvm-dis | grep phi | count 1

define void @foo() {
entry:
        br label %bb6

bb6:            ; preds = %cond_true, %entry
        %j.0 = phi i32 [ 1, %entry ], [ %tmp5, %cond_true ]             ; <i32> [#uses=3]
        %i.0 = phi i32 [ 0, %entry ], [ %j.0, %cond_true ]              ; <i32> [#uses=1]
        %tmp7 = call i32 (...)* @foo2( )                ; <i32> [#uses=1]
        %tmp = icmp ne i32 %tmp7, 0             ; <i1> [#uses=1]
        br i1 %tmp, label %cond_true, label %return

cond_true:              ; preds = %bb6
        %tmp2 = call i32 (...)* @bar( i32 %i.0, i32 %j.0 )              ; <i32> [#uses=0]
        %tmp5 = add i32 %j.0, 1         ; <i32> [#uses=1]
        br label %bb6

return:         ; preds = %bb6
        ret void
}

declare i32 @bar(...)

declare i32 @foo2(...)

