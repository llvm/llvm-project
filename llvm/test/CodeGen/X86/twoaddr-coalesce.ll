; RUN: llc < %s -march=x86 | grep mov | count 2
; rdar://6523745

@"\01LC" = internal constant [4 x i8] c"%d\0A\00"		; <[4 x i8]*> [#uses=1]

define i32 @foo() nounwind {
bb1.thread:
	br label %bb1

bb1:		; preds = %bb1, %bb1.thread
	%i.0.reg2mem.0 = phi i32 [ 0, %bb1.thread ], [ %indvar.next, %bb1 ]		; <i32> [#uses=2]
	%0 = trunc i32 %i.0.reg2mem.0 to i8		; <i8> [#uses=1]
	%1 = sdiv i8 %0, 2		; <i8> [#uses=1]
	%2 = sext i8 %1 to i32		; <i32> [#uses=1]
	%3 = tail call i32 (i8*, ...) @printf(i8* getelementptr ([4 x i8], [4 x i8]* @"\01LC", i32 0, i32 0), i32 %2) nounwind		; <i32> [#uses=0]
	%indvar.next = add i32 %i.0.reg2mem.0, 1		; <i32> [#uses=2]
	%exitcond = icmp eq i32 %indvar.next, 258		; <i1> [#uses=1]
	br i1 %exitcond, label %bb2, label %bb1

bb2:		; preds = %bb1
	ret i32 0
}

declare i32 @printf(i8*, ...) nounwind
