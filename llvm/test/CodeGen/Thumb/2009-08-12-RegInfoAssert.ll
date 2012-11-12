; RUN: llc < %s -mtriple=thumbv6-apple-darwin

	%struct.vorbis_comment = type { i8**, i32*, i32, i8* }
@.str16 = external constant [2 x i8], align 1     ; <[2 x i8]*> [#uses=1]

declare i8* @__strcpy_chk(i8*, i8*, i32) nounwind

declare i8* @__strcat_chk(i8*, i8*, i32) nounwind

define i8* @vorbis_comment_query(%struct.vorbis_comment* nocapture %vc, i8* %tag, i32 %count) nounwind {
entry:
	%0 = alloca i8, i32 undef, align 4        ; <i8*> [#uses=2]
	%1 = call  i8* @__strcpy_chk(i8* %0, i8* %tag, i32 -1) nounwind; <i8*> [#uses=0]
	%2 = call  i8* @__strcat_chk(i8* %0, i8* getelementptr ([2 x i8]* @.str16, i32 0, i32 0), i32 -1) nounwind; <i8*> [#uses=0]
	%3 = getelementptr %struct.vorbis_comment* %vc, i32 0, i32 0; <i8***> [#uses=1]
	br label %bb11

bb6:                                              ; preds = %bb11
	%4 = load i8*** %3, align 4               ; <i8**> [#uses=1]
	%scevgep = getelementptr i8** %4, i32 %8  ; <i8**> [#uses=1]
	%5 = load i8** %scevgep, align 4          ; <i8*> [#uses=1]
	br label %bb3.i

bb3.i:                                            ; preds = %bb3.i, %bb6
	%scevgep7.i = getelementptr i8* %5, i32 0 ; <i8*> [#uses=1]
	%6 = load i8* %scevgep7.i, align 1        ; <i8> [#uses=0]
	br i1 undef, label %bb3.i, label %bb10

bb10:                                             ; preds = %bb3.i
	%7 = add i32 %8, 1                        ; <i32> [#uses=1]
	br label %bb11

bb11:                                             ; preds = %bb10, %entry
	%8 = phi i32 [ %7, %bb10 ], [ 0, %entry ] ; <i32> [#uses=3]
	%9 = icmp sgt i32 undef, %8               ; <i1> [#uses=1]
	br i1 %9, label %bb6, label %bb13

bb13:                                             ; preds = %bb11
	ret i8* null
}
