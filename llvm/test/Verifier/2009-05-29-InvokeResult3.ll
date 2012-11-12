; RUN: not llvm-as < %s >& /dev/null

declare i32 @v()

define i32 @h() {
e:
	%s = invoke i32 @v()
			to label %c unwind label %u		; <i32> [#uses=2]

c:		; preds = %e
	br label %d

d:		; preds = %u, %c
	%p = phi i32 [ %s, %c ], [ %s, %u ]		; <i32> [#uses=1]
	ret i32 %p

u:		; preds = %e
	br label %d
}
