; RUN: opt < %s -analyze -scalar-evolution > %t
; RUN: grep "sext i57 {0,+,199}<%bb> to i64" %t | count 1
; RUN: grep "sext i59 {0,+,199}<%bb> to i64" %t | count 1

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9.6"

define i64 @foo(i64* nocapture %x, i64 %n) nounwind {
entry:
	%t0 = icmp sgt i64 %n, 0		; <i1> [#uses=1]
	br i1 %t0, label %bb, label %return

bb:		; preds = %bb, %entry
	%i.01 = phi i64 [ 0, %entry ], [ %indvar.next, %bb ]		; <i32> [#uses=2]
	%t1 = shl i64 %i.01, 7		; <i32> [#uses=1]
	%t2 = ashr i64 %t1, 7		; <i32> [#uses=1]
	%s1 = shl i64 %i.01, 5		; <i32> [#uses=1]
	%s2 = ashr i64 %s1, 5		; <i32> [#uses=1]
	%t3 = getelementptr i64* %x, i64 %i.01		; <i64*> [#uses=1]
	store i64 0, i64* %t3, align 1
	%indvar.next = add i64 %i.01, 199		; <i32> [#uses=2]
	%exitcond = icmp eq i64 %indvar.next, %n		; <i1> [#uses=1]
	br i1 %exitcond, label %return, label %bb

return:		; preds = %bb, %entry
        %p = phi i64 [ 0, %entry ], [ %t2, %bb ]
        %q = phi i64 [ 0, %entry ], [ %s2, %bb ]
        %v = xor i64 %p, %q
	ret i64 %v
}
