; RUN: llvm-as < %s | opt -indvars
; PR2857

@foo = external global i32		; <i32*> [#uses=1]

define void @test(i32 %n, i32 %arg) {
entry:
	br i1 false, label %bb.nph, label %return

bb.nph:		; preds = %entry
	%0 = load i32* @foo, align 4		; <i32> [#uses=1]
	%1 = sext i32 %0 to i64		; <i64> [#uses=1]
	br label %bb

bb:		; preds = %bb, %bb.nph
	%.in = phi i32 [ %2, %bb ], [ %n, %bb.nph ]		; <i32> [#uses=1]
	%val.02 = phi i64 [ %5, %bb ], [ 0, %bb.nph ]		; <i64> [#uses=2]
	%result.01 = phi i64 [ %4, %bb ], [ 0, %bb.nph ]		; <i64> [#uses=1]
	%2 = add i32 %.in, -1		; <i32> [#uses=2]
	%3 = mul i64 %1, %val.02		; <i64> [#uses=1]
	%4 = add i64 %3, %result.01		; <i64> [#uses=2]
	%5 = add i64 %val.02, 1		; <i64> [#uses=1]
	%6 = icmp sgt i32 %2, 0		; <i1> [#uses=1]
	br i1 %6, label %bb, label %bb3.bb4_crit_edge

bb3.bb4_crit_edge:		; preds = %bb
	%.lcssa = phi i64 [ %4, %bb ]		; <i64> [#uses=0]
	ret void

return:		; preds = %entry
	ret void
}
