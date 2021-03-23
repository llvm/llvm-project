; RUN: %lli %s > /dev/null

define i32 @main() {
	call i32 @mylog( i32 4 )		; <i32>:1 [#uses=0]
	ret i32 0
}

define internal i32 @mylog(i32 %num) {
bb0:
	br label %bb2
bb2:		; preds = %bb2, %bb0
	%reg112 = phi i32 [ 10, %bb2 ], [ 1, %bb0 ]		; <i32> [#uses=1]
	%cann-indvar = phi i32 [ %cann-indvar, %bb2 ], [ 0, %bb0 ]		; <i32> [#uses=1]
	%reg114 = add i32 %reg112, 1		; <i32> [#uses=2]
	%cond222 = icmp slt i32 %reg114, %num		; <i1> [#uses=1]
	br i1 %cond222, label %bb2, label %bb3
bb3:		; preds = %bb2
	ret i32 %reg114
}

