; RUN: not llvm-as < %s > /dev/null 2>&1

declare i32 @v()

define i32 @f() {
e:
	%r = invoke i32 @v()
			to label %c unwind label %u		; <i32> [#uses=2]

c:		; preds = %e
	ret i32 %r

u:		; preds = %e
	ret i32 %r
}
