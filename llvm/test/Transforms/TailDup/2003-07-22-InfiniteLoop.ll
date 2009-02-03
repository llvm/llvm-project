; RUN: llvm-as < %s | opt -tailduplicate -disable-output

define i32 @sum() {
entry:
	br label %loopentry
loopentry:		; preds = %loopentry, %entry
	%i.0 = phi i32 [ 1, %entry ], [ %tmp.3, %loopentry ]		; <i32> [#uses=1]
	%tmp.3 = add i32 %i.0, 1		; <i32> [#uses=1]
	br label %loopentry
}

