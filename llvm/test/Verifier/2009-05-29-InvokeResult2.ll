; RUN: not llvm-as < %s > /dev/null 2>&1

declare i32 @v()

define i32 @g() {
e:
	%s = invoke i32 @v()
			to label %c unwind label %u		; <i32> [#uses=2]

c:		; preds = %e
	ret i32 %s

u:		; preds = %e
	%t = phi i32 [ %s, %e ]		; <i32> [#uses=1]
	ret i32 %t
}
