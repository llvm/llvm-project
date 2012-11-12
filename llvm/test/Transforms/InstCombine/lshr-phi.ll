; RUN: opt < %s -instcombine -S > %t
; RUN: not grep lshr %t
; RUN: grep add %t | count 1

; Instcombine should be able to eliminate the lshr, because only
; bits in the operand which might be non-zero will be shifted
; off the end.

define i32 @hash_string(i8* nocapture %key) nounwind readonly {
entry:
	%t0 = load i8* %key, align 1		; <i8> [#uses=1]
	%t1 = icmp eq i8 %t0, 0		; <i1> [#uses=1]
	br i1 %t1, label %bb2, label %bb

bb:		; preds = %bb, %entry
	%indvar = phi i64 [ 0, %entry ], [ %tmp, %bb ]		; <i64> [#uses=2]
	%k.04 = phi i32 [ 0, %entry ], [ %t8, %bb ]		; <i32> [#uses=2]
	%cp.05 = getelementptr i8* %key, i64 %indvar		; <i8*> [#uses=1]
	%t2 = shl i32 %k.04, 1		; <i32> [#uses=1]
	%t3 = lshr i32 %k.04, 14		; <i32> [#uses=1]
	%t4 = add i32 %t2, %t3		; <i32> [#uses=1]
	%t5 = load i8* %cp.05, align 1		; <i8> [#uses=1]
	%t6 = sext i8 %t5 to i32		; <i32> [#uses=1]
	%t7 = xor i32 %t6, %t4		; <i32> [#uses=1]
	%t8 = and i32 %t7, 16383		; <i32> [#uses=2]
	%tmp = add i64 %indvar, 1		; <i64> [#uses=2]
	%scevgep = getelementptr i8* %key, i64 %tmp		; <i8*> [#uses=1]
	%t9 = load i8* %scevgep, align 1		; <i8> [#uses=1]
	%t10 = icmp eq i8 %t9, 0		; <i1> [#uses=1]
	br i1 %t10, label %bb2, label %bb

bb2:		; preds = %bb, %entry
	%k.0.lcssa = phi i32 [ 0, %entry ], [ %t8, %bb ]		; <i32> [#uses=1]
	ret i32 %k.0.lcssa
}
