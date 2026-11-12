; RUN: llc < %s -mtriple=thumbv6-apple-darwin

	%struct.vorbis_comment = type { ptr, ptr, i32, ptr }
@.str16 = external constant [2 x i8], align 1     ; <ptr> [#uses=1]

declare ptr @__strcpy_chk(ptr, ptr, i32) nounwind

declare ptr @__strcat_chk(ptr, ptr, i32) nounwind

define ptr @vorbis_comment_query(ptr nocapture %vc, ptr %tag, i32 %count) nounwind {
entry:
	%0 = alloca i8, i32 undef, align 4        ; <ptr> [#uses=2]
	%1 = call  ptr @__strcpy_chk(ptr %0, ptr %tag, i32 -1) nounwind; <ptr> [#uses=0]
	%2 = call  ptr @__strcat_chk(ptr %0, ptr @.str16, i32 -1) nounwind; <ptr> [#uses=0]
	%3 = getelementptr %struct.vorbis_comment, ptr %vc, i32 0, i32 0; <ptr> [#uses=1]
	br label %bb11

bb6:                                              ; preds = %bb11
	%4 = load ptr, ptr %3, align 4               ; <ptr> [#uses=1]
	%scevgep = getelementptr ptr, ptr %4, i32 %8  ; <ptr> [#uses=1]
	%5 = load ptr, ptr %scevgep, align 4          ; <ptr> [#uses=1]
	br label %bb3.i

bb3.i:                                            ; preds = %bb3.i, %bb6
	%scevgep7.i = getelementptr i8, ptr %5, i32 0 ; <ptr> [#uses=1]
	%6 = load i8, ptr %scevgep7.i, align 1        ; <i8> [#uses=0]
	br i1 undef, label %bb3.i, label %bb10

bb10:                                             ; preds = %bb3.i
	%7 = add i32 %8, 1                        ; <i32> [#uses=1]
	br label %bb11

bb11:                                             ; preds = %bb10, %entry
	%8 = phi i32 [ %7, %bb10 ], [ 0, %entry ] ; <i32> [#uses=3]
	%9 = icmp sgt i32 undef, %8               ; <i1> [#uses=1]
	br i1 %9, label %bb6, label %bb13

bb13:                                             ; preds = %bb11
	ret ptr null
}
