; RUN: opt < %s -simplify-libcalls -disable-output
; PR4645

define i32 @main() {
entry:
	br label %if.then

lor.lhs.false:		; preds = %while.body
	br i1 undef, label %if.then, label %for.cond

if.then:		; preds = %lor.lhs.false, %while.body
	call void @exit(i32 1)
	br label %for.cond

for.cond:		; preds = %for.end, %if.then, %lor.lhs.false
	%j.0 = phi i32 [ %inc47, %for.end ], [ 0, %if.then ], [ 0, %lor.lhs.false ]		; <i32> [#uses=1]
	unreachable

for.end:		; preds = %for.cond20
	%inc47 = add i32 %j.0, 1		; <i32> [#uses=1]
	br label %for.cond
}

declare void @exit(i32)
