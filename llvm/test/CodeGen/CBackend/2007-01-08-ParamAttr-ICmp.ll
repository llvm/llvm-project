; For PR1099
; RUN: llvm-as < %s | llc -march=c | grep {(llvm_cbe_tmp2 == llvm_cbe_b_2e_0_2e_0_2e_val)}

target datalayout = "e-p:32:32"
target triple = "i686-apple-darwin8"
        %struct.Connector = type { i16, i16, i8, i8, %struct.Connector*, i8* }


define i1 @prune_match_entry_2E_ce(%struct.Connector* %a, i16 %b.0.0.val) {
newFuncRoot:
        br label %entry.ce

cond_next.exitStub:             ; preds = %entry.ce
        ret i1 true

entry.return_crit_edge.exitStub:                ; preds = %entry.ce
        ret i1 false

entry.ce:               ; preds = %newFuncRoot
        %tmp1 = getelementptr %struct.Connector* %a, i32 0, i32 0                ; <i16*> [#uses=1]
        %tmp2 = load i16* %tmp1           ; <i16> [#uses=1]
        %tmp3 = icmp eq i16 %tmp2, %b.0.0.val             ; <i1> [#uses=1]
        br i1 %tmp3, label %cond_next.exitStub, label %entry.return_crit_edge.exitStub
}


