; RUN: llvm-as < %s | opt -tailduplicate | \
; RUN:   llc -march=x86 -o %t -f
; RUN: grep {\\\<je\\\>} %t
; RUN: not grep jmp %t
; END.
; This should have no unconditional jumps in it.  The C source is:

;void foo(int c, int* P) {
;  if (c & 1)  P[0] = 1;
;  if (c & 2)  P[1] = 1;
;  if (c & 4)  P[2] = 1;
;  if (c & 8)  P[3] = 1;
;}

define void @foo(i32 %c, i32* %P) {
entry:
	%tmp1 = and i32 %c, 1		; <i32> [#uses=1]
	%tmp1.upgrd.1 = icmp eq i32 %tmp1, 0		; <i1> [#uses=1]
	br i1 %tmp1.upgrd.1, label %cond_next, label %cond_true
cond_true:		; preds = %entry
	store i32 1, i32* %P
	br label %cond_next
cond_next:		; preds = %cond_true, %entry
	%tmp5 = and i32 %c, 2		; <i32> [#uses=1]
	%tmp5.upgrd.2 = icmp eq i32 %tmp5, 0		; <i1> [#uses=1]
	br i1 %tmp5.upgrd.2, label %cond_next10, label %cond_true6
cond_true6:		; preds = %cond_next
	%tmp8 = getelementptr i32* %P, i32 1		; <i32*> [#uses=1]
	store i32 1, i32* %tmp8
	br label %cond_next10
cond_next10:		; preds = %cond_true6, %cond_next
	%tmp13 = and i32 %c, 4		; <i32> [#uses=1]
	%tmp13.upgrd.3 = icmp eq i32 %tmp13, 0		; <i1> [#uses=1]
	br i1 %tmp13.upgrd.3, label %cond_next18, label %cond_true14
cond_true14:		; preds = %cond_next10
	%tmp16 = getelementptr i32* %P, i32 2		; <i32*> [#uses=1]
	store i32 1, i32* %tmp16
	br label %cond_next18
cond_next18:		; preds = %cond_true14, %cond_next10
	%tmp21 = and i32 %c, 8		; <i32> [#uses=1]
	%tmp21.upgrd.4 = icmp eq i32 %tmp21, 0		; <i1> [#uses=1]
	br i1 %tmp21.upgrd.4, label %return, label %cond_true22
cond_true22:		; preds = %cond_next18
	%tmp24 = getelementptr i32* %P, i32 3		; <i32*> [#uses=1]
	store i32 1, i32* %tmp24
	ret void
return:		; preds = %cond_next18
	ret void
}
