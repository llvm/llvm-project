; RUN: llc < %s -march=x86 > %t
; RUN: not grep neg %t
; RUN: not grep sub.*esp %t
; RUN: not grep esi %t
; RUN: not grep push %t

; This corresponds to:
;int t(int a, int b) {
;  while (a != b) {
;    if (a > b)
;      a -= b;
;    else
;      b -= a;
;  }
;  return a;
;}


define i32 @t(i32 %a, i32 %b) nounwind {
entry:
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
