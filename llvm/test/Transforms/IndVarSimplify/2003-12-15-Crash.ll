; RUN: opt < %s -indvars -disable-output 
define void @_ZN17CoinFactorization7cleanupEv() {
entry:
        br i1 false, label %loopexit.14, label %cond_continue.3

cond_continue.3:                ; preds = %entry
        ret void

loopexit.14:            ; preds = %entry
        %tmp.738 = sub i32 0, 0         ; <i32> [#uses=1]
        br i1 undef, label %no_exit.15.preheader, label %loopexit.15

no_exit.15.preheader:           ; preds = %loopexit.14
        br label %no_exit.15

no_exit.15:             ; preds = %no_exit.15, %no_exit.15.preheader
        %highC.0 = phi i32 [ %tmp.738, %no_exit.15.preheader ], [ %dec.0, %no_exit.15 ]         ; <i32> [#uses=1]
        %dec.0 = add i32 %highC.0, -1           ; <i32> [#uses=1]
        br i1 undef, label %no_exit.15, label %loopexit.15

loopexit.15:            ; preds = %no_exit.15, %loopexit.14
        ret void
}

