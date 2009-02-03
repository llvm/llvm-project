; PR 1334
; RUN: llvm-as < %s | opt -loop-unroll -disable-output

define void @sal__math_float_manipulator_7__math__joint_array_dcv_ops__Omultiply__3([6 x float]* %agg.result) {
entry:
	%tmp282911 = zext i8 0 to i32		; <i32> [#uses=1]
	br label %cond_next
cond_next:		; preds = %cond_next, %entry
	%indvar = phi i8 [ 0, %entry ], [ %indvar.next, %cond_next ]		; <i8> [#uses=1]
	%indvar.next = add i8 %indvar, 1		; <i8> [#uses=2]
	%exitcond = icmp eq i8 %indvar.next, 7		; <i1> [#uses=1]
	br i1 %exitcond, label %bb27, label %cond_next
bb27:		; preds = %cond_next
	%tmp282911.lcssa = phi i32 [ %tmp282911, %cond_next ]		; <i32> [#uses=0]
	ret void
}

