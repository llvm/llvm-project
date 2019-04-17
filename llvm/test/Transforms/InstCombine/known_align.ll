; RUN: opt < %s -instcombine -S | grep "align 1"
; END.

	%struct.p = type <{ i8, i32 }>
@t = global %struct.p <{ i8 1, i32 10 }>		; <%struct.p*> [#uses=1]
@u = weak global %struct.p zeroinitializer		; <%struct.p*> [#uses=1]

define i32 @main() {
entry:
	%retval = alloca i32, align 4		; <i32*> [#uses=2]
	%tmp = alloca i32, align 4		; <i32*> [#uses=2]
	%tmp1 = alloca i32, align 4		; <i32*> [#uses=3]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	%tmp3 = load i32, i32* getelementptr (%struct.p, %struct.p* @t, i32 0, i32 1), align 1		; <i32> [#uses=1]
	store i32 %tmp3, i32* %tmp1, align 4
	%tmp5 = load i32, i32* %tmp1, align 4		; <i32> [#uses=1]
	store i32 %tmp5, i32* getelementptr (%struct.p, %struct.p* @u, i32 0, i32 1), align 1
	%tmp6 = load i32, i32* %tmp1, align 4		; <i32> [#uses=1]
	store i32 %tmp6, i32* %tmp, align 4
	%tmp7 = load i32, i32* %tmp, align 4		; <i32> [#uses=1]
	store i32 %tmp7, i32* %retval, align 4
	br label %return

return:		; preds = %entry
	%retval8 = load i32, i32* %retval		; <i32> [#uses=1]
	ret i32 %retval8
}
