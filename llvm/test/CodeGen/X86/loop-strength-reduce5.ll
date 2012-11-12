; RUN: llc < %s -march=x86 | grep inc | count 1

@X = weak global i16 0		; <i16*> [#uses=1]
@Y = weak global i16 0		; <i16*> [#uses=1]

define void @foo(i32 %N) nounwind {
entry:
	%tmp1019 = icmp sgt i32 %N, 0		; <i1> [#uses=1]
	br i1 %tmp1019, label %bb, label %return

bb:		; preds = %bb, %entry
	%i.014.0 = phi i32 [ 0, %entry ], [ %indvar.next, %bb ]		; <i32> [#uses=2]
	%tmp1 = trunc i32 %i.014.0 to i16		; <i16> [#uses=2]
	store volatile i16 %tmp1, i16* @X, align 2
	%tmp34 = shl i16 %tmp1, 2		; <i16> [#uses=1]
	store volatile i16 %tmp34, i16* @Y, align 2
	%indvar.next = add i32 %i.014.0, 1		; <i32> [#uses=2]
	%exitcond = icmp eq i32 %indvar.next, %N		; <i1> [#uses=1]
	br i1 %exitcond, label %return, label %bb

return:		; preds = %bb, %entry
	ret void
}
