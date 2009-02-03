; RUN: llvm-as < %s | opt -analyze -scalar-evolution |& not grep smax
; PR2261

@lut = common global [256 x i8] zeroinitializer, align 32		; <[256 x i8]*> [#uses=1]

define void @foo(i32 %count, i32* %srcptr, i32* %dstptr) nounwind  {
entry:
	icmp sgt i32 %count, 0		; <i1>:0 [#uses=1]
	br i1 %0, label %bb.nph, label %return

bb.nph:		; preds = %entry
	br label %bb

bb:		; preds = %bb1, %bb.nph
	%j.01 = phi i32 [ %8, %bb1 ], [ 0, %bb.nph ]		; <i32> [#uses=1]
	load i32* %srcptr, align 4		; <i32>:1 [#uses=2]
	and i32 %1, 255		; <i32>:2 [#uses=1]
	and i32 %1, -256		; <i32>:3 [#uses=1]
	getelementptr [256 x i8]* @lut, i32 0, i32 %2		; <i8*>:4 [#uses=1]
	load i8* %4, align 1		; <i8>:5 [#uses=1]
	zext i8 %5 to i32		; <i32>:6 [#uses=1]
	or i32 %6, %3		; <i32>:7 [#uses=1]
	store i32 %7, i32* %dstptr, align 4
	add i32 %j.01, 1		; <i32>:8 [#uses=2]
	br label %bb1

bb1:		; preds = %bb
	icmp slt i32 %8, %count		; <i1>:9 [#uses=1]
	br i1 %9, label %bb, label %bb1.return_crit_edge

bb1.return_crit_edge:		; preds = %bb1
	br label %return

return:		; preds = %bb1.return_crit_edge, %entry
	ret void
}
