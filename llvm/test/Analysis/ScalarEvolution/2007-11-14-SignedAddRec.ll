; RUN: opt < %s -indvars -S | grep printd | grep 1206807378
; PR1798

declare void @printd(i32)

define i32 @test() {
entry:
	br label %bb6

bb:		; preds = %bb6
	%tmp3 = add i32 %x.0, %i.0		; <i32> [#uses=1]
	%tmp5 = add i32 %i.0, 1		; <i32> [#uses=1]
	br label %bb6

bb6:		; preds = %bb, %entry
	%i.0 = phi i32 [ 0, %entry ], [ %tmp5, %bb ]		; <i32> [#uses=3]
	%x.0 = phi i32 [ 0, %entry ], [ %tmp3, %bb ]		; <i32> [#uses=3]
	%tmp8 = icmp slt i32 %i.0, 123456789		; <i1> [#uses=1]
	br i1 %tmp8, label %bb, label %bb10

bb10:		; preds = %bb6
	call void @printd(i32 %x.0)
	ret i32 0
}
