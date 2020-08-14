; RUN: opt < %s -loop-rotate -licm -simple-loop-unswitch -disable-output

define i32 @main(i32 %argc, i8** %argv) {
entry:
	br label %bb7

bb7:		; preds = %bb7, %entry
	%tmp54 = icmp slt i32 0, 2000000		; <i1> [#uses=1]
	br i1 %tmp54, label %bb7, label %bb56

bb56:		; preds = %bb7
	ret i32 0
}
