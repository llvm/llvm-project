; RUN: opt < %s -passes='loop(indvars),instcombine' -S | FileCheck %s
;
; Test that -passes=indvars can reduce variable stride IVs.  If it can reduce variable
; stride iv's, it will make %iv. and %m.0.0 isomorphic to each other without
; cycles, allowing the tmp.21 subtraction to be eliminated.

define void @vnum_test8(i32* %data) {
entry:
        %tmp.1 = getelementptr i32, i32* %data, i32 3                ; <i32*> [#uses=1]
        %tmp.2 = load i32, i32* %tmp.1               ; <i32> [#uses=2]
        %tmp.4 = getelementptr i32, i32* %data, i32 4                ; <i32*> [#uses=1]
        %tmp.5 = load i32, i32* %tmp.4               ; <i32> [#uses=2]
        %tmp.8 = getelementptr i32, i32* %data, i32 2                ; <i32*> [#uses=1]
        %tmp.9 = load i32, i32* %tmp.8               ; <i32> [#uses=3]
        %tmp.125 = icmp sgt i32 %tmp.2, 0               ; <i1> [#uses=1]
        br i1 %tmp.125, label %no_exit.preheader, label %return

no_exit.preheader:              ; preds = %entry
        %tmp.16 = getelementptr i32, i32* %data, i32 %tmp.9          ; <i32*> [#uses=1]
        br label %no_exit

; CHECK: store i32 0
no_exit:                ; preds = %no_exit, %no_exit.preheader
        %iv.ui = phi i32 [ 0, %no_exit.preheader ], [ %iv..inc.ui, %no_exit ]           ; <i32> [#uses=1]
        %iv. = phi i32 [ %tmp.5, %no_exit.preheader ], [ %iv..inc, %no_exit ]           ; <i32> [#uses=2]
        %m.0.0 = phi i32 [ %tmp.5, %no_exit.preheader ], [ %tmp.24, %no_exit ]          ; <i32> [#uses=2]
        store i32 2, i32* %tmp.16
        %tmp.21 = sub i32 %m.0.0, %iv.          ; <i32> [#uses=1]
        store i32 %tmp.21, i32* %data
        %tmp.24 = add i32 %m.0.0, %tmp.9                ; <i32> [#uses=1]
        %iv..inc = add i32 %tmp.9, %iv.         ; <i32> [#uses=1]
        %iv..inc.ui = add i32 %iv.ui, 1         ; <i32> [#uses=2]
        %iv..inc1 = bitcast i32 %iv..inc.ui to i32              ; <i32> [#uses=1]
        %tmp.12 = icmp slt i32 %iv..inc1, %tmp.2                ; <i1> [#uses=1]
        br i1 %tmp.12, label %no_exit, label %return.loopexit

return.loopexit:                ; preds = %no_exit
        br label %return

return:         ; preds = %return.loopexit, %entry
        ret void
}

