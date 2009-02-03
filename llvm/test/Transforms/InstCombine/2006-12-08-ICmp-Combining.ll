; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:    grep {%bothcond =}

define i1 @Doit_bb(i32 %i.0) {
bb:
        %tmp = icmp sgt i32 %i.0, 0             ; <i1> [#uses=1]
        %tmp.not = xor i1 %tmp, true            ; <i1> [#uses=1]
        %tmp2 = icmp sgt i32 %i.0, 8            ; <i1> [#uses=1]
        %bothcond = or i1 %tmp.not, %tmp2               ; <i1> [#uses=1]
        br i1 %bothcond, label %exitTrue, label %exitFalse

exitTrue:               ; preds = %bb
        ret i1 true

exitFalse:              ; preds = %bb
        ret i1 false
}

