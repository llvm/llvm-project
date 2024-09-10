; RUN: llc < %s -mtriple=arm-apple-darwin -mattr=+v6,+vfp2

@accum = external global { double, double }		; <ptr> [#uses=1]
@.str = external constant [4 x i8]		; <ptr> [#uses=1]

define i32 @main() {
entry:
	br label %bb74.i
bb74.i:		; preds = %bb88.i, %bb74.i, %entry
	br i1 false, label %bb88.i, label %bb74.i
bb88.i:		; preds = %bb74.i
	br i1 false, label %mandel.exit, label %bb74.i
mandel.exit:		; preds = %bb88.i
	%tmp2 = load volatile double, ptr @accum, align 8		; <double> [#uses=1]
	%tmp23 = fptosi double %tmp2 to i32		; <i32> [#uses=1]
	%tmp5 = tail call i32 (ptr, ...) @printf( ptr @.str, i32 %tmp23 )		; <i32> [#uses=0]
	ret i32 0
}

declare i32 @printf(ptr, ...)
