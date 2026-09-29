; RUN: llc < %s

define void @QRiterate(i32 %p.1, double %tmp.212) {
entry:
        %tmp.184 = icmp sgt i32 %p.1, 0         ; <i1> [#uses=1]
        br i1 %tmp.184, label %shortcirc_next.1, label %shortcirc_done.1

shortcirc_next.1:               ; preds = %shortcirc_done.1, %entry
        %tmp.213 = fcmp une double %tmp.212, 0.000000e+00               ; <i1> [#uses=1]
        br label %shortcirc_done.1

shortcirc_done.1:               ; preds = %shortcirc_next.1, %entry
        %val.1 = phi i1 [ false, %entry ], [ %tmp.213, %shortcirc_next.1 ]              ; <i1> [#uses=1]
        br i1 %val.1, label %shortcirc_next.1, label %exit.1

exit.1:         ; preds = %shortcirc_done.1
        ret void
}

