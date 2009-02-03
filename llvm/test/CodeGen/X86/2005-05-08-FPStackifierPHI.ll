; RUN: llvm-as < %s | llc -march=x86 -mcpu=generic
; Make sure LLC doesn't crash in the stackifier due to FP PHI nodes.

define void @radfg_() {
entry:
        br i1 false, label %no_exit.16.preheader, label %loopentry.0
loopentry.0:            ; preds = %entry
        ret void
no_exit.16.preheader:           ; preds = %entry
        br label %no_exit.16
no_exit.16:             ; preds = %no_exit.16, %no_exit.16.preheader
        br i1 false, label %loopexit.16.loopexit, label %no_exit.16
loopexit.16.loopexit:           ; preds = %no_exit.16
        br label %no_exit.18
no_exit.18:             ; preds = %loopexit.20, %loopexit.16.loopexit
        %tmp.882 = add float 0.000000e+00, 0.000000e+00         ; <float> [#uses=2]
        br i1 false, label %loopexit.19, label %no_exit.19.preheader
no_exit.19.preheader:           ; preds = %no_exit.18
        ret void
loopexit.19:            ; preds = %no_exit.18
        br i1 false, label %loopexit.20, label %no_exit.20
no_exit.20:             ; preds = %loopexit.21, %loopexit.19
        %ai2.1122.tmp.3 = phi float [ %tmp.958, %loopexit.21 ], [ %tmp.882, %loopexit.19 ]              ; <float> [#uses=1]
        %tmp.950 = mul float %tmp.882, %ai2.1122.tmp.3          ; <float> [#uses=1]
        %tmp.951 = sub float 0.000000e+00, %tmp.950             ; <float> [#uses=1]
        %tmp.958 = add float 0.000000e+00, 0.000000e+00         ; <float> [#uses=1]
        br i1 false, label %loopexit.21, label %no_exit.21.preheader
no_exit.21.preheader:           ; preds = %no_exit.20
        ret void
loopexit.21:            ; preds = %no_exit.20
        br i1 false, label %loopexit.20, label %no_exit.20
loopexit.20:            ; preds = %loopexit.21, %loopexit.19
        %ar2.1124.tmp.2 = phi float [ 0.000000e+00, %loopexit.19 ], [ %tmp.951, %loopexit.21 ]          ; <float> [#uses=0]
        br i1 false, label %loopexit.18.loopexit, label %no_exit.18
loopexit.18.loopexit:           ; preds = %loopexit.20
        ret void
}

