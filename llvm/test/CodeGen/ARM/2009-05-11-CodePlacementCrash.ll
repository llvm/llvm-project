; RUN: llc -mtriple=arm-eabi %s -o /dev/null

	%struct.List = type { ptr, i32 }
@Node5 = external constant %struct.List		; <ptr> [#uses=1]
@"\01LC" = external constant [7 x i8]		; <ptr> [#uses=1]

define i32 @main() nounwind {
entry:
	br label %bb

bb:		; preds = %bb3, %entry
	%CurL.02 = phi ptr [ @Node5, %entry ], [ %2, %bb3 ]		; <ptr> [#uses=1]
	%PrevL.01 = phi ptr [ null, %entry ], [ %CurL.02, %bb3 ]		; <ptr> [#uses=1]
	%0 = icmp eq ptr %PrevL.01, null		; <i1> [#uses=1]
	br i1 %0, label %bb3, label %bb1

bb1:		; preds = %bb
	br label %bb3

bb3:		; preds = %bb1, %bb
	%iftmp.0.0 = phi i32 [ 0, %bb1 ], [ -1, %bb ]		; <i32> [#uses=1]
	%1 = tail call i32 (ptr, ...) @printf(ptr @"\01LC", i32 0, i32 %iftmp.0.0) nounwind		; <i32> [#uses=0]
	%2 = load ptr, ptr null, align 4		; <ptr> [#uses=2]
	%phitmp = icmp eq ptr %2, null		; <i1> [#uses=1]
	br i1 %phitmp, label %bb5, label %bb

bb5:		; preds = %bb3
	ret i32 0
}

declare i32 @printf(ptr nocapture, ...) nounwind
