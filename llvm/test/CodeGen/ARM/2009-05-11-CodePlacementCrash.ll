; RUN: llc < %s -march=arm
	%struct.List = type { %struct.List*, i32 }
@Node5 = external constant %struct.List		; <%struct.List*> [#uses=1]
@"\01LC" = external constant [7 x i8]		; <[7 x i8]*> [#uses=1]

define i32 @main() nounwind {
entry:
	br label %bb

bb:		; preds = %bb3, %entry
	%CurL.02 = phi %struct.List* [ @Node5, %entry ], [ %2, %bb3 ]		; <%struct.List*> [#uses=1]
	%PrevL.01 = phi %struct.List* [ null, %entry ], [ %CurL.02, %bb3 ]		; <%struct.List*> [#uses=1]
	%0 = icmp eq %struct.List* %PrevL.01, null		; <i1> [#uses=1]
	br i1 %0, label %bb3, label %bb1

bb1:		; preds = %bb
	br label %bb3

bb3:		; preds = %bb1, %bb
	%iftmp.0.0 = phi i32 [ 0, %bb1 ], [ -1, %bb ]		; <i32> [#uses=1]
	%1 = tail call i32 (i8*, ...)* @printf(i8* getelementptr ([7 x i8]* @"\01LC", i32 0, i32 0), i32 0, i32 %iftmp.0.0) nounwind		; <i32> [#uses=0]
	%2 = load %struct.List** null, align 4		; <%struct.List*> [#uses=2]
	%phitmp = icmp eq %struct.List* %2, null		; <i1> [#uses=1]
	br i1 %phitmp, label %bb5, label %bb

bb5:		; preds = %bb3
	ret i32 0
}

declare i32 @printf(i8* nocapture, ...) nounwind
