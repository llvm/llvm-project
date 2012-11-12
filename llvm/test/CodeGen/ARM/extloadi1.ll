; RUN: llc < %s -march=arm
@handler_installed.6144.b = external global i1          ; <i1*> [#uses=1]

define void @__mf_sigusr1_respond() {
entry:
        %tmp8.b = load i1* @handler_installed.6144.b            ; <i1> [#uses=1]
        br i1 false, label %cond_true7, label %cond_next

cond_next:              ; preds = %entry
        br i1 %tmp8.b, label %bb, label %cond_next3

cond_next3:             ; preds = %cond_next
        ret void

bb:             ; preds = %cond_next
        ret void

cond_true7:             ; preds = %entry
        ret void
}
