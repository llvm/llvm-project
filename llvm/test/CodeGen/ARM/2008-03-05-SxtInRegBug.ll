; RUN: llc < %s -march=arm -mattr=+v6 | not grep 255

define i32 @main(i32 %argc, i8** %argv) {
entry:
	br label %bb1
bb1:		; preds = %entry
	%tmp3.i.i = load i8* null, align 1		; <i8> [#uses=1]
	%tmp4.i.i = icmp slt i8 %tmp3.i.i, 0		; <i1> [#uses=1]
	br i1 %tmp4.i.i, label %bb2, label %bb3
bb2:		; preds = %bb1
	ret i32 1
bb3:		; preds = %bb1
	ret i32 0
}
