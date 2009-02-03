; RUN: llvm-as < %s | opt -simplifycfg -disable-output
; PR 2800

define void @foo() {
start:
	%tmp = call i1 @bar( )		; <i1> [#uses=4]
	br i1 %tmp, label %brtrue, label %brfalse

brtrue:		; preds = %start
	%tmpnew = and i1 %tmp, %tmp		; <i1> [#uses=1]
	br label %brfalse

brfalse:		; preds = %brtrue, %start
	%andandtmp.0 = phi i1 [ %tmp, %start ], [ %tmpnew, %brtrue ]		; <i1> [#uses=0]
	ret void
}

declare i1 @bar()
