; RUN: opt < %s -passes=instcombine -disable-output
target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f128:64:128"
target triple = "powerpc-unknown-linux-gnu"

define i32 @test() {
entry:
	%tmp50.i17 = icmp slt i32 0, 4		; <i1> [#uses=1]
	br i1 %tmp50.i17, label %bb.i, label %calculateColorSpecificBlackLevel.exit

bb.i:		; preds = %entry
	br label %bb51.i.i

bb27.i.i:		; preds = %bb51.i.i
	%tmp31.i.i = load i16, i16* null, align 2		; <i16> [#uses=2]
	%tmp35.i.i = icmp ult i16 %tmp31.i.i, 1		; <i1> [#uses=1]
	%tmp41.i.i = icmp ugt i16 %tmp31.i.i, -1		; <i1> [#uses=1]
	%bothcond.i.i = or i1 %tmp35.i.i, %tmp41.i.i		; <i1> [#uses=1]
	%bothcond1.i.i = zext i1 %bothcond.i.i to i32		; <i32> [#uses=1]
	%tmp46.i.i = xor i32 %bothcond1.i.i, 1		; <i32> [#uses=1]
	%count.0.i.i = add i32 %count.1.i.i, %tmp46.i.i		; <i32> [#uses=1]
	%tmp50.i.i = add i32 %x.0.i.i, 2		; <i32> [#uses=1]
	br label %bb51.i.i

bb51.i.i:		; preds = %bb27.i.i, %bb.i
	%count.1.i.i = phi i32 [ %count.0.i.i, %bb27.i.i ], [ 0, %bb.i ]		; <i32> [#uses=1]
	%x.0.i.i = phi i32 [ %tmp50.i.i, %bb27.i.i ], [ 0, %bb.i ]		; <i32> [#uses=2]
	%tmp54.i.i = icmp slt i32 %x.0.i.i, 0		; <i1> [#uses=1]
	br i1 %tmp54.i.i, label %bb27.i.i, label %bb57.i.i

bb57.i.i:		; preds = %bb51.i.i
	ret i32 0

calculateColorSpecificBlackLevel.exit:		; preds = %entry
	ret i32 undef
}
