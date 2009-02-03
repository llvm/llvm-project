; RUN: llvm-as < %s | opt -indvars -loop-deletion -simplifycfg | llvm-dis | not grep br
;
; Testcase distilled from 256.bzip2

define i32 @main() {
entry:
        br label %loopentry

loopentry:              ; preds = %loopentry, %entry
        %indvar1 = phi i32 [ 0, %entry ], [ %indvar.next2, %loopentry ]         ; <i32> [#uses=1]
        %h.0 = phi i32 [ %tmp.2, %loopentry ], [ 4, %entry ]            ; <i32> [#uses=1]
        %tmp.1 = mul i32 %h.0, 3                ; <i32> [#uses=1]
        %tmp.2 = add i32 %tmp.1, 1              ; <i32> [#uses=2]
        %indvar.next2 = add i32 %indvar1, 1             ; <i32> [#uses=2]
        %exitcond3 = icmp ne i32 %indvar.next2, 4               ; <i1> [#uses=1]
        br i1 %exitcond3, label %loopentry, label %loopexit

loopexit:               ; preds = %loopentry
        ret i32 %tmp.2
}

