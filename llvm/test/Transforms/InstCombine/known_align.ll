; RUN: opt < %s -passes=instcombine -S | grep "align 1"
; END.

	%struct.p = type <{ i8, i32 }>
@t = global %struct.p <{ i8 1, i32 10 }>		; <ptr> [#uses=1]
@u = weak global %struct.p zeroinitializer		; <ptr> [#uses=1]

define i32 @main() {
entry:
	%retval = alloca i32, align 4		; <ptr> [#uses=2]
	%tmp = alloca i32, align 4		; <ptr> [#uses=2]
	%tmp1 = alloca i32, align 4		; <ptr> [#uses=3]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	%tmp3 = load i32, ptr getelementptr (%struct.p, ptr @t, i32 0, i32 1), align 1		; <i32> [#uses=1]
	store i32 %tmp3, ptr %tmp1, align 4
	%tmp5 = load i32, ptr %tmp1, align 4		; <i32> [#uses=1]
	store i32 %tmp5, ptr getelementptr (%struct.p, ptr @u, i32 0, i32 1), align 1
	%tmp6 = load i32, ptr %tmp1, align 4		; <i32> [#uses=1]
	store i32 %tmp6, ptr %tmp, align 4
	%tmp7 = load i32, ptr %tmp, align 4		; <i32> [#uses=1]
	store i32 %tmp7, ptr %retval, align 4
	br label %return

return:		; preds = %entry
	%retval8 = load i32, ptr %retval		; <i32> [#uses=1]
	ret i32 %retval8
}
