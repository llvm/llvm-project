; RUN: llc < %s -mtriple=x86_64-- > %t
; RUN: grep inc %t | count 1
; RUN: not grep incw %t

@X = common global i16 0		; <i16*> [#uses=1]

define i32 @foo(i32 %N) nounwind {
entry:
	%0 = icmp sgt i32 %N, 0		; <i1> [#uses=1]
	br i1 %0, label %bb, label %return

bb:		; preds = %bb, %entry
	%i.03 = phi i32 [ 0, %entry ], [ %indvar.next, %bb ]		; <i32> [#uses=2]
	%1 = trunc i32 %i.03 to i16		; <i16> [#uses=1]
	store volatile i16 %1, i16* @X, align 2
	%indvar.next = add i32 %i.03, 1		; <i32> [#uses=2]
	%exitcond = icmp eq i32 %indvar.next, %N		; <i1> [#uses=1]
	br i1 %exitcond, label %return, label %bb

return:		; preds = %bb, %entry
        %h = phi i32 [ 0, %entry ], [ %indvar.next, %bb ]
	ret i32 %h
}
