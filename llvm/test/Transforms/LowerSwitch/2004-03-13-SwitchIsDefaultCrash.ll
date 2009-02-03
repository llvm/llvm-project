; RUN: llvm-as < %s | opt -lowerswitch -disable-output

define void @solve() {
entry:
	%targetBlock = call i16 @solve_code( )		; <i16> [#uses=1]
	br label %codeReplTail
then.1:		; preds = %codeReplTail
	ret void
loopexit.0:		; preds = %codeReplTail
	ret void
codeReplTail:		; preds = %codeReplTail, %entry
	switch i16 %targetBlock, label %codeReplTail [
		 i16 0, label %loopexit.0
		 i16 1, label %then.1
	]
}

declare i16 @solve_code()

