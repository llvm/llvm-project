; RUN: llvm-as < %s | opt -loopsimplify -licm -disable-output

; This is PR306

define void @NormalizeCoeffsVecFFE() {
entry:
	br label %loopentry.0
loopentry.0:		; preds = %no_exit.0, %entry
	br i1 false, label %loopentry.1, label %no_exit.0
no_exit.0:		; preds = %loopentry.0
	br i1 false, label %loopentry.0, label %loopentry.1
loopentry.1:		; preds = %no_exit.1, %no_exit.0, %loopentry.0
	br i1 false, label %no_exit.1, label %loopexit.1
no_exit.1:		; preds = %loopentry.1
	%tmp.43 = icmp eq i16 0, 0		; <i1> [#uses=1]
	br i1 %tmp.43, label %loopentry.1, label %loopexit.1
loopexit.1:		; preds = %no_exit.1, %loopentry.1
	ret void
}

