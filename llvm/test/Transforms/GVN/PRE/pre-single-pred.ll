; RUN: opt < %s -gvn -enable-load-pre -S | FileCheck %s

@p = external global i32
define i32 @f(i32 %n) nounwind {
; CHECK: entry:
; CHECK-NEXT: %0 = load i32, i32* @p
entry:
	br label %for.cond

for.cond:		; preds = %for.inc, %entry
	%i.0 = phi i32 [ 0, %entry ], [ %indvar.next, %for.inc ]		; <i32> [#uses=2]
	%cmp = icmp slt i32 %i.0, %n		; <i1> [#uses=1]
	br i1 %cmp, label %for.body, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:		; preds = %for.cond
	br label %for.end

; The load of @p should be hoisted into the entry block.
; CHECK: for.body:
; CHECK-NEXT: %dec = add i32 %tmp3, -1
for.body:		; preds = %for.cond
	%tmp3 = load i32, i32* @p		; <i32> [#uses=1]
	%dec = add i32 %tmp3, -1		; <i32> [#uses=2]
	store i32 %dec, i32* @p
	%cmp6 = icmp slt i32 %dec, 0		; <i1> [#uses=1]
	br i1 %cmp6, label %for.body.for.end_crit_edge, label %for.inc

; CHECK: for.body.for.end_crit_edge:
for.body.for.end_crit_edge:		; preds = %for.body
	br label %for.end

for.inc:		; preds = %for.body
	%indvar.next = add i32 %i.0, 1		; <i32> [#uses=1]
	br label %for.cond

for.end:		; preds = %for.body.for.end_crit_edge, %for.cond.for.end_crit_edge
	%tmp9 = load i32, i32* @p		; <i32> [#uses=1]
	ret i32 %tmp9
}
