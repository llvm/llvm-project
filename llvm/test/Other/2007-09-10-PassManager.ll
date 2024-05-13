; RUN: opt < %s -passes=simple-loop-unswitch,indvars -disable-output
; Require SCEV before LCSSA.
define void @foo() {
entry:
	%i = alloca i32, align 4		; <ptr> [#uses=5]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store i32 0, ptr %i, align 4
	br label %bb3

bb:		; preds = %bb3
	%tmp = load i32, ptr %i, align 4		; <i32> [#uses=1]
	call void @bar( i32 %tmp )
	%tmp1 = load i32, ptr %i, align 4		; <i32> [#uses=1]
	%tmp2 = add i32 %tmp1, 1		; <i32> [#uses=1]
	store i32 %tmp2, ptr %i, align 4
	br label %bb3

bb3:		; preds = %bb, %entry
	%tmp4 = load i32, ptr %i, align 4		; <i32> [#uses=1]
	%tmp5 = icmp sle i32 %tmp4, 9		; <i1> [#uses=1]
	%tmp56 = zext i1 %tmp5 to i8		; <i8> [#uses=1]
	%toBool = icmp ne i8 %tmp56, 0		; <i1> [#uses=1]
	br i1 %toBool, label %bb, label %bb7

bb7:		; preds = %bb3
	br label %return

return:		; preds = %bb7
	ret void
}

declare void @bar(i32)
