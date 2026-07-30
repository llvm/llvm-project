; RUN: opt < %s -passes=indvars -disable-output

define void @.outPlank_21() {
entry:
        br i1 false, label %loopexit.0, label %no_exit.0

no_exit.0:              ; preds = %entry
        ret void

loopexit.0:             ; preds = %entry
        br i1 false, label %no_exit.1, label %loopexit.1

no_exit.1:              ; preds = %loopexit.2, %loopexit.0
        %i.0.0 = phi i32 [ %inc, %loopexit.2 ], [ 0, %loopexit.0 ]              ; <i32> [#uses=1]
        br i1 false, label %loopexit.2, label %no_exit.2

no_exit.2:              ; preds = %no_exit.1
        ret void

loopexit.2:             ; preds = %no_exit.1
        %inc = add i32 %i.0.0, 1                ; <i32> [#uses=1]
        br i1 false, label %no_exit.1, label %loopexit.1

loopexit.1:             ; preds = %loopexit.2, %loopexit.0
        ret void
}

