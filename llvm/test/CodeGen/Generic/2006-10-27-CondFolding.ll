; RUN: llc < %s 

define void @start_pass_huff(i32 %gather_statistics) {
entry:
        %tmp = icmp eq i32 %gather_statistics, 0                ; <i1> [#uses=1]
        br i1 false, label %cond_next22, label %bb166

cond_next22:            ; preds = %entry
        %bothcond = and i1 false, %tmp          ; <i1> [#uses=1]
        br i1 %bothcond, label %bb34, label %bb46

bb34:           ; preds = %cond_next22
        ret void

bb46:           ; preds = %cond_next22
        ret void

bb166:          ; preds = %entry
        ret void
}

