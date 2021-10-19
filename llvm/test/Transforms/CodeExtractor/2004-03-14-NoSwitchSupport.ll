; RUN: opt < %s -loop-simplify -loop-extract-single -disable-output

define void @ab() {
entry:
        br label %codeReplTail

then.1:         ; preds = %codeReplTail
        br label %loopentry.1

loopentry.1:            ; preds = %no_exit.1, %then.1
        br i1 false, label %no_exit.1, label %loopexit.0.loopexit1

no_exit.1:              ; preds = %loopentry.1
        br label %loopentry.1

loopexit.0.loopexit:            ; preds = %codeReplTail
        ret void

loopexit.0.loopexit1:           ; preds = %loopentry.1
        ret void

codeReplTail:           ; preds = %codeReplTail, %entry
        switch i16 0, label %codeReplTail [
                 i16 0, label %loopexit.0.loopexit
                 i16 1, label %then.1
        ]
}

