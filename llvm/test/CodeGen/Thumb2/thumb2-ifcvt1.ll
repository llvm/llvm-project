; RUN: llc < %s -mtriple=thumbv7-apple-darwin | FileCheck %s
; RUN: llc < %s -mtriple=thumbv7-apple-darwin -arm-default-it | FileCheck %s
; RUN: llc < %s -mtriple=thumbv8 -arm-no-restrict-it | FileCheck %s
; RUN: llc < %s -mtriple=thumbv8 -arm-no-restrict-it -enable-tail-merge=0 | FileCheck %s
define i32 @t1(i32 %a, i32 %b, i32 %c, i32 %d) nounwind {
; CHECK-LABEL: t1:
; CHECK: ittt ne
; CHECK: cmpne
; CHECK: addne
; CHECK: bxne lr
	switch i32 %c, label %cond_next [
		 i32 1, label %cond_true
		 i32 7, label %cond_true
	]

cond_true:
	%tmp12 = add i32 %a, 1
	%tmp1518 = add i32 %tmp12, %b
	ret i32 %tmp1518

cond_next:
	%tmp15 = add i32 %b, %a
	ret i32 %tmp15
}

define i32 @t2(i32 %a, i32 %b) nounwind {
entry:
; CHECK-LABEL: t2:
; CHECK: ite {{gt|le}}
; CHECK-DAG: suble
; CHECK-DAG: subgt
	%tmp1434 = icmp eq i32 %a, %b		; <i1> [#uses=1]
	br i1 %tmp1434, label %bb17, label %bb.outer

bb.outer:		; preds = %cond_false, %entry
	%b_addr.021.0.ph = phi i32 [ %b, %entry ], [ %tmp10, %cond_false ]		; <i32> [#uses=5]
	%a_addr.026.0.ph = phi i32 [ %a, %entry ], [ %a_addr.026.0, %cond_false ]		; <i32> [#uses=1]
	br label %bb

bb:		; preds = %cond_true, %bb.outer
	%indvar = phi i32 [ 0, %bb.outer ], [ %indvar.next, %cond_true ]		; <i32> [#uses=2]
	%tmp. = sub i32 0, %b_addr.021.0.ph		; <i32> [#uses=1]
	%tmp.40 = mul i32 %indvar, %tmp.		; <i32> [#uses=1]
	%a_addr.026.0 = add i32 %tmp.40, %a_addr.026.0.ph		; <i32> [#uses=6]
	%tmp3 = icmp sgt i32 %a_addr.026.0, %b_addr.021.0.ph		; <i1> [#uses=1]
	br i1 %tmp3, label %cond_true, label %cond_false

cond_true:		; preds = %bb
	%tmp7 = sub i32 %a_addr.026.0, %b_addr.021.0.ph		; <i32> [#uses=2]
	%tmp1437 = icmp eq i32 %tmp7, %b_addr.021.0.ph		; <i1> [#uses=1]
	%indvar.next = add i32 %indvar, 1		; <i32> [#uses=1]
	br i1 %tmp1437, label %bb17, label %bb

cond_false:		; preds = %bb
	%tmp10 = sub i32 %b_addr.021.0.ph, %a_addr.026.0		; <i32> [#uses=2]
	%tmp14 = icmp eq i32 %a_addr.026.0, %tmp10		; <i1> [#uses=1]
	br i1 %tmp14, label %bb17, label %bb.outer

bb17:		; preds = %cond_false, %cond_true, %entry
	%a_addr.026.1 = phi i32 [ %a, %entry ], [ %tmp7, %cond_true ], [ %a_addr.026.0, %cond_false ]		; <i32> [#uses=1]
	ret i32 %a_addr.026.1
}

define i32 @t2_nomerge(i32 %a, i32 %b) nounwind {
entry:
; CHECK-LABEL: t2_nomerge:
; CHECK-NOT: ite {{gt|le}}
; CHECK-NOT: suble
; CHECK-NOT: subgt
	%tmp1434 = icmp eq i32 %a, %b		; <i1> [#uses=1]
	br i1 %tmp1434, label %bb17, label %bb.outer

bb.outer:		; preds = %cond_false, %entry
	%b_addr.021.0.ph = phi i32 [ %b, %entry ], [ %tmp10, %cond_false ]		; <i32> [#uses=5]
	%a_addr.026.0.ph = phi i32 [ %a, %entry ], [ %a_addr.026.0, %cond_false ]		; <i32> [#uses=1]
	br label %bb

bb:		; preds = %cond_true, %bb.outer
	%indvar = phi i32 [ 0, %bb.outer ], [ %indvar.next, %cond_true ]		; <i32> [#uses=2]
	%tmp. = sub i32 0, %b_addr.021.0.ph		; <i32> [#uses=1]
	%tmp.40 = mul i32 %indvar, %tmp.		; <i32> [#uses=1]
	%a_addr.026.0 = add i32 %tmp.40, %a_addr.026.0.ph		; <i32> [#uses=6]
	%tmp3 = icmp sgt i32 %a_addr.026.0, %b_addr.021.0.ph		; <i1> [#uses=1]
	br i1 %tmp3, label %cond_true, label %cond_false

cond_true:		; preds = %bb
	%tmp7 = sub i32 %a_addr.026.0, %b_addr.021.0.ph		; <i32> [#uses=2]
	%tmp1437 = icmp eq i32 %tmp7, %b_addr.021.0.ph		; <i1> [#uses=1]
	%indvar.next = add i32 %indvar, 1		; <i32> [#uses=1]
	br i1 %tmp1437, label %bb17, label %bb

cond_false:		; preds = %bb
	%tmp10 = sub i32 %b_addr.021.0.ph, %a_addr.026.0		; <i32> [#uses=2]
	%tmp14 = icmp eq i32 %b_addr.021.0.ph, %tmp10		; <i1> [#uses=1]
	br i1 %tmp14, label %bb17, label %bb.outer

bb17:		; preds = %cond_false, %cond_true, %entry
	%a_addr.026.1 = phi i32 [ %a, %entry ], [ %tmp7, %cond_true ], [ %a_addr.026.0, %cond_false ]		; <i32> [#uses=1]
	ret i32 %a_addr.026.1
}

@x = external global i32*		; <i32**> [#uses=1]

define void @foo(i32 %a) nounwind {
entry:
	%tmp = load i32*, i32** @x		; <i32*> [#uses=1]
	store i32 %a, i32* %tmp
	ret void
}

define void @t3(i32 %a, i32 %b) nounwind {
entry:
; CHECK-LABEL: t3:
; CHECK: it lt
; CHECK-NEXT: bxlt lr
; CHECK: mov r0, r1
; CHECK: bl  {{_?}}foo
	%tmp1 = icmp sgt i32 %a, 10		; <i1> [#uses=1]
	br i1 %tmp1, label %cond_true, label %UnifiedReturnBlock

cond_true:		; preds = %entry
	call void @foo( i32 %b )
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}
