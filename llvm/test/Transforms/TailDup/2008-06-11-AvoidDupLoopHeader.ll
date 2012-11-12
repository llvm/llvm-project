; RUN: opt < %s -tailduplicate -taildup-threshold=3 -stats -disable-output 2>&1 | not grep tailduplicate
; XFAIL: *

define i32 @foo(i32 %l) nounwind  {
entry:
	%cond = icmp eq i32 %l, 1		; <i1> [#uses=1]
	br i1 %cond, label %bb, label %bb9

bb:		; preds = %entry
	br label %bb9

bb5:		; preds = %bb9
	%tmp7 = call i32 (...)* @bar( i32 %x.0 ) nounwind 		; <i32> [#uses=1]
	br label %bb9

bb9:		; preds = %bb5, %bb, %entry
	%x.0 = phi i32 [ 0, %entry ], [ %tmp7, %bb5 ], [ 1525, %bb ]		; <i32> [#uses=2]
	%l_addr.0 = phi i32 [ %l, %entry ], [ %tmp11, %bb5 ], [ %l, %bb ]		; <i32> [#uses=1]
	%tmp11 = add i32 %l_addr.0, -1		; <i32> [#uses=2]
	%tmp13 = icmp eq i32 %tmp11, -1		; <i1> [#uses=1]
	br i1 %tmp13, label %bb15, label %bb5

bb15:		; preds = %bb9
	ret i32 %x.0
}

declare i32 @bar(...)
