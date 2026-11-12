; RUN: opt < %s -passes=instcombine -S | grep "ashr"
; PR1499

define void @av_cmp_q_cond_true(ptr %retval, ptr %tmp9, ptr %tmp10) {
newFuncRoot:
	br label %cond_true

return.exitStub:		; preds = %cond_true
	ret void

cond_true:		; preds = %newFuncRoot
	%tmp30 = load i64, ptr %tmp10		; <i64> [#uses=1]
	%.cast = zext i32 63 to i64		; <i64> [#uses=1]
	%tmp31 = ashr i64 %tmp30, %.cast		; <i64> [#uses=1]
	%tmp3132 = trunc i64 %tmp31 to i32		; <i32> [#uses=1]
	%tmp33 = or i32 %tmp3132, 1		; <i32> [#uses=1]
	store i32 %tmp33, ptr %tmp9
	%tmp34 = load i32, ptr %tmp9		; <i32> [#uses=1]
	store i32 %tmp34, ptr %retval
	br label %return.exitStub
}

