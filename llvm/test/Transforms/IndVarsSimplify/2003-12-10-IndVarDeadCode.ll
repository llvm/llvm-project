; The induction variable canonicalization pass shouldn't leave dead
; instructions laying around!
;
; RUN: llvm-as < %s | opt -indvars | llvm-dis | \
; RUN:   not grep {#uses=0}

define i32 @mul(i32 %x, i32 %y) {
entry:
        br label %tailrecurse

tailrecurse:            ; preds = %endif, %entry
        %accumulator.tr = phi i32 [ %x, %entry ], [ %tmp.9, %endif ]            ; <i32> [#uses=2]
        %y.tr = phi i32 [ %y, %entry ], [ %tmp.8, %endif ]              ; <i32> [#uses=2]
        %tmp.1 = icmp eq i32 %y.tr, 0           ; <i1> [#uses=1]
        br i1 %tmp.1, label %return, label %endif

endif:          ; preds = %tailrecurse
        %tmp.8 = add i32 %y.tr, -1              ; <i32> [#uses=1]
        %tmp.9 = add i32 %accumulator.tr, %x            ; <i32> [#uses=1]
        br label %tailrecurse

return:         ; preds = %tailrecurse
        ret i32 %accumulator.tr
}

