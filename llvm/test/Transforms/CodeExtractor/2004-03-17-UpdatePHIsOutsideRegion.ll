; RUN: llvm-as < %s | opt -loop-extract -disable-output

define void @maketree() {
entry:
        br i1 false, label %no_exit.1, label %loopexit.0

no_exit.1:              ; preds = %endif, %expandbox.entry, %entry
        br i1 false, label %endif, label %expandbox.entry

expandbox.entry:                ; preds = %no_exit.1
        br i1 false, label %loopexit.1, label %no_exit.1

endif:          ; preds = %no_exit.1
        br i1 false, label %loopexit.1, label %no_exit.1

loopexit.1:             ; preds = %endif, %expandbox.entry
        %ic.i.0.0.4 = phi i32 [ 0, %expandbox.entry ], [ 0, %endif ]            ; <i32> [#uses=0]
        ret void

loopexit.0:             ; preds = %entry
        ret void
}

