; These tests ensure that we can compute the trip count of various forms of
; loops.  If the trip count of the loop is computable, then we will know what
; the exit value of the loop will be for some value, allowing us to substitute
; it directly into users outside of the loop, making the loop dead.
;
; RUN: llvm-as < %s | opt -indvars -loop-deletion -simplifycfg | llvm-dis | not grep br

define i32 @linear_setne() {
entry:
	br label %loop

loop:		; preds = %loop, %entry
	%i = phi i32 [ 0, %entry ], [ %i.next, %loop ]		; <i32> [#uses=3]
	%i.next = add i32 %i, 1		; <i32> [#uses=1]
	%c = icmp ne i32 %i, 100		; <i1> [#uses=1]
	br i1 %c, label %loop, label %loopexit

loopexit:		; preds = %loop
	ret i32 %i
}

define i32 @linear_setne_2() {
entry:
	br label %loop

loop:		; preds = %loop, %entry
	%i = phi i32 [ 0, %entry ], [ %i.next, %loop ]		; <i32> [#uses=3]
	%i.next = add i32 %i, 2		; <i32> [#uses=1]
	%c = icmp ne i32 %i, 100		; <i1> [#uses=1]
	br i1 %c, label %loop, label %loopexit

loopexit:		; preds = %loop
	ret i32 %i
}

define i32 @linear_setne_overflow() {
entry:
	br label %loop

loop:		; preds = %loop, %entry
	%i = phi i32 [ 1024, %entry ], [ %i.next, %loop ]		; <i32> [#uses=3]
	%i.next = add i32 %i, 1024		; <i32> [#uses=1]
	%c = icmp ne i32 %i, 0		; <i1> [#uses=1]
	br i1 %c, label %loop, label %loopexit

loopexit:		; preds = %loop
	ret i32 %i
}

define i32 @linear_setlt() {
entry:
	br label %loop

loop:		; preds = %loop, %entry
	%i = phi i32 [ 0, %entry ], [ %i.next, %loop ]		; <i32> [#uses=3]
	%i.next = add i32 %i, 1		; <i32> [#uses=1]
	%c = icmp slt i32 %i, 100		; <i1> [#uses=1]
	br i1 %c, label %loop, label %loopexit

loopexit:		; preds = %loop
	ret i32 %i
}

define i32 @quadratic_setlt() {
entry:
	br label %loop

loop:		; preds = %loop, %entry
	%i = phi i32 [ 7, %entry ], [ %i.next, %loop ]		; <i32> [#uses=4]
	%i.next = add i32 %i, 3		; <i32> [#uses=1]
	%i2 = mul i32 %i, %i		; <i32> [#uses=1]
	%c = icmp slt i32 %i2, 1000		; <i1> [#uses=1]
	br i1 %c, label %loop, label %loopexit

loopexit:		; preds = %loop
	ret i32 %i
}

define i32 @chained() {
entry:
	br label %loop

loop:		; preds = %loop, %entry
	%i = phi i32 [ 0, %entry ], [ %i.next, %loop ]		; <i32> [#uses=3]
	%i.next = add i32 %i, 1		; <i32> [#uses=1]
	%c = icmp ne i32 %i, 100		; <i1> [#uses=1]
	br i1 %c, label %loop, label %loopexit

loopexit:		; preds = %loop
	br label %loop2

loop2:		; preds = %loop2, %loopexit
	%j = phi i32 [ %i, %loopexit ], [ %j.next, %loop2 ]		; <i32> [#uses=3]
	%j.next = add i32 %j, 1		; <i32> [#uses=1]
	%c2 = icmp ne i32 %j, 200		; <i1> [#uses=1]
	br i1 %c2, label %loop2, label %loopexit2

loopexit2:		; preds = %loop2
	ret i32 %j
}
