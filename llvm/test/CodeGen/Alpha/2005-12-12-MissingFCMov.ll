; This shouldn't crash
; RUN: llvm-as < %s | llc -march=alpha

@.str_4 = external global [44 x i8]             ; <[44 x i8]*> [#uses=0]

declare void @printf(i32, ...)

define void @main() {
entry:
        %tmp.11861 = icmp slt i64 0, 1          ; <i1> [#uses=1]
        %tmp.19466 = icmp slt i64 0, 1          ; <i1> [#uses=1]
        %tmp.21571 = icmp slt i64 0, 1          ; <i1> [#uses=1]
        %tmp.36796 = icmp slt i64 0, 1          ; <i1> [#uses=1]
        br i1 %tmp.11861, label %loopexit.2, label %no_exit.2

no_exit.2:              ; preds = %entry
        ret void

loopexit.2:             ; preds = %entry
        br i1 %tmp.19466, label %loopexit.3, label %no_exit.3.preheader

no_exit.3.preheader:            ; preds = %loopexit.2
        ret void

loopexit.3:             ; preds = %loopexit.2
        br i1 %tmp.21571, label %no_exit.6, label %no_exit.4

no_exit.4:              ; preds = %loopexit.3
        ret void

no_exit.6:              ; preds = %no_exit.6, %loopexit.3
        %tmp.30793 = icmp sgt i64 0, 0          ; <i1> [#uses=1]
        br i1 %tmp.30793, label %loopexit.6, label %no_exit.6

loopexit.6:             ; preds = %no_exit.6
        %Z.1 = select i1 %tmp.36796, double 1.000000e+00, double 0x3FEFFF7CEDE74EAE; <double> [#uses=2]
        tail call void (i32, ...)* @printf( i32 0, i64 0, i64 0, i64 0, double 1.000000e+00, double 1.000000e+00, double %Z.1, double %Z.1 )
        ret void
}

