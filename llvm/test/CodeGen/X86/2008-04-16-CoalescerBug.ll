; RUN: llc < %s -march=x86

define void @Hubba(i8* %saveunder, i32 %firstBlob, i32 %select) nounwind  {
entry:
	br i1 false, label %bb53.us, label %bb53
bb53.us:		; preds = %bb94.us, %bb53.us, %entry
	switch i8 1, label %bb71.us [
		 i8 0, label %bb53.us
		 i8 1, label %bb94.us
	]
bb94.us:		; preds = %bb71.us, %bb53.us
	%result.0.us = phi i32 [ %tmp93.us, %bb71.us ], [ 0, %bb53.us ]		; <i32> [#uses=2]
	%tmp101.us = lshr i32 %result.0.us, 3		; <i32> [#uses=1]
	%result.0163.us = trunc i32 %result.0.us to i16		; <i16> [#uses=2]
	shl i16 %result.0163.us, 7		; <i16>:0 [#uses=1]
	%tmp106.us = and i16 %0, -1024		; <i16> [#uses=1]
	shl i16 %result.0163.us, 2		; <i16>:1 [#uses=1]
	%tmp109.us = and i16 %1, -32		; <i16> [#uses=1]
	%tmp111112.us = trunc i32 %tmp101.us to i16		; <i16> [#uses=1]
	%tmp110.us = or i16 %tmp109.us, %tmp111112.us		; <i16> [#uses=1]
	%tmp113.us = or i16 %tmp110.us, %tmp106.us		; <i16> [#uses=1]
	store i16 %tmp113.us, i16* null, align 2
	br label %bb53.us
bb71.us:		; preds = %bb53.us
	%tmp80.us = load i8, i8* null, align 1		; <i8> [#uses=1]
	%tmp8081.us = zext i8 %tmp80.us to i32		; <i32> [#uses=1]
	%tmp87.us = mul i32 %tmp8081.us, 0		; <i32> [#uses=1]
	%tmp92.us = add i32 0, %tmp87.us		; <i32> [#uses=1]
	%tmp93.us = udiv i32 %tmp92.us, 255		; <i32> [#uses=1]
	br label %bb94.us
bb53:		; preds = %entry
	ret void
}
