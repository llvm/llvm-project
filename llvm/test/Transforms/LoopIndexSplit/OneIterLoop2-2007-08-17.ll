; Loop is elimianted. Save last value assignment.
; RUN: llvm-as < %s | opt -loop-index-split -disable-output -stats |& \
; RUN: grep "loop-index-split" | count 1

	%struct.anon = type { i32 }
@S1 = external global i32		; <i32*> [#uses=1]
@W1 = external global i32		; <i32*> [#uses=1]
@Y = weak global [100 x %struct.anon] zeroinitializer, align 32		; <[100 x %struct.anon]*> [#uses=1]
@ti = external global i32		; <i32*> [#uses=1]
@T2 = external global [100 x [100 x i32]]		; <[100 x [100 x i32]]*> [#uses=1]
@d = external global i32		; <i32*> [#uses=1]
@T1 = external global i32		; <i32*> [#uses=2]
@N1 = external global i32		; <i32*> [#uses=2]

define void @foo() {
entry:
	%tmp = load i32* @S1, align 4		; <i32> [#uses=4]
	%tmp266 = load i32* @N1, align 4		; <i32> [#uses=1]
	%tmp288 = icmp ult i32 %tmp, %tmp266		; <i1> [#uses=1]
	br i1 %tmp288, label %bb.preheader, label %return

bb.preheader:		; preds = %entry
	%tmp1 = load i32* @W1, align 4		; <i32> [#uses=1]
	%tmp13 = load i32* @ti, align 4		; <i32> [#uses=1]
	%tmp18 = load i32* @d, align 4		; <i32> [#uses=1]
	%tmp26 = load i32* @N1, align 4		; <i32> [#uses=2]
	%T1.promoted = load i32* @T1		; <i32> [#uses=1]
	%tmp2 = add i32 %tmp, 1		; <i32> [#uses=2]
	%tmp4 = icmp ugt i32 %tmp2, %tmp26		; <i1> [#uses=1]
	%umax = select i1 %tmp4, i32 %tmp2, i32 %tmp26		; <i32> [#uses=1]
	%tmp5 = sub i32 0, %tmp		; <i32> [#uses=1]
	%tmp6 = add i32 %umax, %tmp5		; <i32> [#uses=1]
	br label %bb

bb:		; preds = %bb25, %bb.preheader
	%indvar = phi i32 [ 0, %bb.preheader ], [ %indvar.next, %bb25 ]		; <i32> [#uses=2]
	%T1.tmp.1 = phi i32 [ %T1.promoted, %bb.preheader ], [ %T1.tmp.0, %bb25 ]		; <i32> [#uses=3]
	%tj.01.0 = add i32 %indvar, %tmp		; <i32> [#uses=3]
	%tmp24 = add i32 %tj.01.0, 1		; <i32> [#uses=1]
	%tmp3 = icmp eq i32 %tmp24, %tmp1		; <i1> [#uses=1]
	br i1 %tmp3, label %cond_true, label %bb25

cond_true:		; preds = %bb
	%tmp7 = getelementptr [100 x %struct.anon]* @Y, i32 0, i32 %tj.01.0, i32 0		; <i32*> [#uses=1]
	%tmp8 = load i32* %tmp7, align 4		; <i32> [#uses=1]
	%tmp9 = icmp sgt i32 %tmp8, 0		; <i1> [#uses=1]
	br i1 %tmp9, label %cond_true12, label %bb25

cond_true12:		; preds = %cond_true
	%tmp16 = getelementptr [100 x [100 x i32]]* @T2, i32 0, i32 %tmp13, i32 %tj.01.0		; <i32*> [#uses=1]
	%tmp17 = load i32* %tmp16, align 4		; <i32> [#uses=1]
	%tmp19 = mul i32 %tmp18, %tmp17		; <i32> [#uses=1]
	%tmp21 = add i32 %tmp19, %T1.tmp.1		; <i32> [#uses=1]
	br label %bb25

bb25:		; preds = %cond_true12, %cond_true, %bb
	%T1.tmp.0 = phi i32 [ %T1.tmp.1, %bb ], [ %T1.tmp.1, %cond_true ], [ %tmp21, %cond_true12 ]		; <i32> [#uses=2]
	%indvar.next = add i32 %indvar, 1		; <i32> [#uses=2]
	%exitcond = icmp ne i32 %indvar.next, %tmp6		; <i1> [#uses=1]
	br i1 %exitcond, label %bb, label %return.loopexit

return.loopexit:		; preds = %bb25
	%T1.tmp.0.lcssa = phi i32 [ %T1.tmp.0, %bb25 ]		; <i32> [#uses=1]
	store i32 %T1.tmp.0.lcssa, i32* @T1
	br label %return

return:		; preds = %return.loopexit, %entry
	ret void
}
