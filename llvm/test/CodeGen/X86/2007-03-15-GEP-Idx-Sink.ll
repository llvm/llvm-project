; RUN: llc < %s -mtriple=i686-darwin | FileCheck %s

define void @foo(ptr %buf, i32 %size, i32 %col, ptr %p) nounwind {
entry:
; CHECK-LABEL: @foo
; CHECK: push
; CHECK: push
; CHECK: push
; CHECK-NOT: push

	icmp sgt i32 %size, 0		; <i1>:0 [#uses=1]
	br i1 %0, label %bb.preheader, label %return

bb.preheader:		; preds = %entry
	%tmp5.sum72 = add i32 %col, 7		; <i32> [#uses=1]
	%tmp5.sum71 = add i32 %col, 5		; <i32> [#uses=1]
	%tmp5.sum70 = add i32 %col, 3		; <i32> [#uses=1]
	%tmp5.sum69 = add i32 %col, 2		; <i32> [#uses=1]
	%tmp5.sum68 = add i32 %col, 1		; <i32> [#uses=1]
	%tmp5.sum66 = add i32 %col, 4		; <i32> [#uses=1]
	%tmp5.sum = add i32 %col, 6		; <i32> [#uses=1]
	br label %bb

bb:		; preds = %bb, %bb.preheader
	%i.073.0 = phi i32 [ 0, %bb.preheader ], [ %indvar.next, %bb ]		; <i32> [#uses=3]
	%p_addr.076.0.rec = mul i32 %i.073.0, 9		; <i32> [#uses=9]
	%p_addr.076.0 = getelementptr i8, ptr %p, i32 %p_addr.076.0.rec		; <ptr> [#uses=1]
	%tmp2 = getelementptr ptr, ptr %buf, i32 %i.073.0		; <ptr> [#uses=1]
	%tmp3 = load ptr, ptr %tmp2		; <ptr> [#uses=8]
	%tmp5 = getelementptr i8, ptr %tmp3, i32 %col		; <ptr> [#uses=1]
	%tmp7 = load i8, ptr %p_addr.076.0		; <i8> [#uses=1]
	store i8 %tmp7, ptr %tmp5
	%p_addr.076.0.sum93 = add i32 %p_addr.076.0.rec, 1		; <i32> [#uses=1]
	%tmp11 = getelementptr i8, ptr %p, i32 %p_addr.076.0.sum93		; <ptr> [#uses=1]
	%tmp13 = load i8, ptr %tmp11		; <i8> [#uses=1]
	%tmp15 = getelementptr i8, ptr %tmp3, i32 %tmp5.sum72		; <ptr> [#uses=1]
	store i8 %tmp13, ptr %tmp15
	%p_addr.076.0.sum92 = add i32 %p_addr.076.0.rec, 2		; <i32> [#uses=1]
	%tmp17 = getelementptr i8, ptr %p, i32 %p_addr.076.0.sum92		; <ptr> [#uses=1]
	%tmp19 = load i8, ptr %tmp17		; <i8> [#uses=1]
	%tmp21 = getelementptr i8, ptr %tmp3, i32 %tmp5.sum71		; <ptr> [#uses=1]
	store i8 %tmp19, ptr %tmp21
	%p_addr.076.0.sum91 = add i32 %p_addr.076.0.rec, 3		; <i32> [#uses=1]
	%tmp23 = getelementptr i8, ptr %p, i32 %p_addr.076.0.sum91		; <ptr> [#uses=1]
	%tmp25 = load i8, ptr %tmp23		; <i8> [#uses=1]
	%tmp27 = getelementptr i8, ptr %tmp3, i32 %tmp5.sum70		; <ptr> [#uses=1]
	store i8 %tmp25, ptr %tmp27
	%p_addr.076.0.sum90 = add i32 %p_addr.076.0.rec, 4		; <i32> [#uses=1]
	%tmp29 = getelementptr i8, ptr %p, i32 %p_addr.076.0.sum90		; <ptr> [#uses=1]
	%tmp31 = load i8, ptr %tmp29		; <i8> [#uses=1]
	%tmp33 = getelementptr i8, ptr %tmp3, i32 %tmp5.sum69		; <ptr> [#uses=2]
	store i8 %tmp31, ptr %tmp33
	%p_addr.076.0.sum89 = add i32 %p_addr.076.0.rec, 5		; <i32> [#uses=1]
	%tmp35 = getelementptr i8, ptr %p, i32 %p_addr.076.0.sum89		; <ptr> [#uses=1]
	%tmp37 = load i8, ptr %tmp35		; <i8> [#uses=1]
	%tmp39 = getelementptr i8, ptr %tmp3, i32 %tmp5.sum68		; <ptr> [#uses=1]
	store i8 %tmp37, ptr %tmp39
	%p_addr.076.0.sum88 = add i32 %p_addr.076.0.rec, 6		; <i32> [#uses=1]
	%tmp41 = getelementptr i8, ptr %p, i32 %p_addr.076.0.sum88		; <ptr> [#uses=1]
	%tmp43 = load i8, ptr %tmp41		; <i8> [#uses=1]
	store i8 %tmp43, ptr %tmp33
	%p_addr.076.0.sum87 = add i32 %p_addr.076.0.rec, 7		; <i32> [#uses=1]
	%tmp47 = getelementptr i8, ptr %p, i32 %p_addr.076.0.sum87		; <ptr> [#uses=1]
	%tmp49 = load i8, ptr %tmp47		; <i8> [#uses=1]
	%tmp51 = getelementptr i8, ptr %tmp3, i32 %tmp5.sum66		; <ptr> [#uses=1]
	store i8 %tmp49, ptr %tmp51
	%p_addr.076.0.sum = add i32 %p_addr.076.0.rec, 8		; <i32> [#uses=1]
	%tmp53 = getelementptr i8, ptr %p, i32 %p_addr.076.0.sum		; <ptr> [#uses=1]
	%tmp55 = load i8, ptr %tmp53		; <i8> [#uses=1]
	%tmp57 = getelementptr i8, ptr %tmp3, i32 %tmp5.sum		; <ptr> [#uses=1]
	store i8 %tmp55, ptr %tmp57
	%indvar.next = add i32 %i.073.0, 1		; <i32> [#uses=2]
	icmp eq i32 %indvar.next, %size		; <i1>:1 [#uses=1]
	br i1 %1, label %return, label %bb

return:		; preds = %bb, %entry
	ret void
}
