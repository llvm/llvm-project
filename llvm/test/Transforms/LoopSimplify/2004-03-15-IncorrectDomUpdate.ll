; RUN: opt < %s -loop-simplify -licm -disable-output
define void @main() {
entry:
	br i1 false, label %Out, label %loop
loop:		; preds = %loop, %entry
	%LI = icmp sgt i32 0, 0		; <i1> [#uses=1]
	br i1 %LI, label %loop, label %Out
Out:		; preds = %loop, %entry
	ret void
}

