; RUN: llvm-as < %s | opt -tailduplicate -disable-output

define void @ab() {
entry:
	br label %loopentry.5
loopentry.5:		; preds = %no_exit.5, %entry
	%poscnt.1 = phi i64 [ 0, %entry ], [ %tmp.289, %no_exit.5 ]		; <i64> [#uses=1]
	%tmp.289 = ashr i64 %poscnt.1, 1		; <i64> [#uses=1]
	br i1 false, label %no_exit.5, label %loopexit.5
no_exit.5:		; preds = %loopentry.5
	br label %loopentry.5
loopexit.5:		; preds = %loopentry.5
	ret void
}

