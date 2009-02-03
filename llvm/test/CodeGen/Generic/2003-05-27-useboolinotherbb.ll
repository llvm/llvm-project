; RUN: llvm-as < %s | llc

define void @QRiterate(double %tmp.212) {
        %tmp.213 = fcmp une double %tmp.212, 0.000000e+00               ; <i1> [#uses=1]
        br label %shortcirc_next.1

shortcirc_next.1:               ; preds = %shortcirc_next.1, %0
        br i1 %tmp.213, label %shortcirc_next.1, label %exit.1

exit.1:         ; preds = %shortcirc_next.1
        ret void
}

