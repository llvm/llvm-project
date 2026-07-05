; RUN: opt < %s -passes=adce -S | FileCheck %s
; RUN: opt < %s -passes=adce -adce-remove-loops -S | FileCheck %s

%FILE = type { i32, ptr, ptr, i8, i8, i32, i32, i32 }
	%spec_fd_t = type { i32, i32, i32, ptr }
@__iob = external global [20 x %FILE]		; <ptr> [#uses=1]
@dbglvl = global i32 4		; <ptr> [#uses=3]
@spec_fd = external global [3 x %spec_fd_t]		; <ptr> [#uses=4]
@.LC9 = internal global [34 x i8] c"spec_read: fd=%d, > MAX_SPEC_FD!\0A\00"		; <ptr> [#uses=1]
@.LC10 = internal global [4 x i8] c"EOF\00"		; <ptr> [#uses=1]
@.LC11 = internal global [4 x i8] c"%d\0A\00"		; <ptr> [#uses=1]
@.LC12 = internal global [17 x i8] c"spec_getc: %d = \00"		; <ptr> [#uses=1]

declare i32 @fprintf(ptr, ptr, ...)

declare void @exit(i32)

declare i32 @remove(ptr)

declare i32 @fputc(i32, ptr)

declare i32 @fwrite(ptr, i32, i32, ptr)

declare void @perror(ptr)

define i32 @spec_getc(i32 %fd) {
	%reg109 = load i32, ptr @dbglvl		; <i32> [#uses=1]
	%cond266 = icmp sle i32 %reg109, 4		; <i1> [#uses=1]
; CHECKL br label %bb3
	br i1 %cond266, label %bb3, label %bb2

bb2:		; preds = %0
	%cast273 = getelementptr [17 x i8], ptr @.LC12, i64 0, i64 0		; <ptr> [#uses=0]
	br label %bb3

bb3:		; preds = %bb2, %0
	%cond267 = icmp sle i32 %fd, 3		; <i1> [#uses=1]
	br i1 %cond267, label %bb5, label %bb4

bb4:		; preds = %bb3
	%reg111 = getelementptr [20 x %FILE], ptr @__iob, i64 0, i64 1, i32 3		; <ptr> [#uses=1]
	%cast274 = getelementptr [34 x i8], ptr @.LC9, i64 0, i64 0		; <ptr> [#uses=0]
	call void @exit( i32 1 )
	br label %UnifiedExitNode

bb5:		; preds = %bb3
	%reg107-idxcast1 = sext i32 %fd to i64		; <i64> [#uses=2]
	%reg107-idxcast2 = sext i32 %fd to i64		; <i64> [#uses=1]
	%reg1311 = getelementptr [3 x %spec_fd_t], ptr @spec_fd, i64 0, i64 %reg107-idxcast2		; <ptr> [#uses=1]
	%idx1 = getelementptr [3 x %spec_fd_t], ptr @spec_fd, i64 0, i64 %reg107-idxcast1, i32 2		; <ptr> [#uses=1]
	%reg1321 = load i32, ptr %idx1		; <i32> [#uses=3]
	%idx2 = getelementptr %spec_fd_t, ptr %reg1311, i64 0, i32 1		; <ptr> [#uses=1]
	%reg1331 = load i32, ptr %idx2		; <i32> [#uses=1]
	%cond270 = icmp slt i32 %reg1321, %reg1331		; <i1> [#uses=1]
	br i1 %cond270, label %bb9, label %bb6

bb6:		; preds = %bb5
	%reg134 = load i32, ptr @dbglvl		; <i32> [#uses=1]
	%cond271 = icmp sle i32 %reg134, 4		; <i1> [#uses=1]
; CHECK: br label %bb8
	br i1 %cond271, label %bb8, label %bb7

bb7:		; preds = %bb6
	%cast277 = getelementptr [4 x i8], ptr @.LC10, i64 0, i64 0		; <ptr> [#uses=0]
	br label %bb8

bb8:		; preds = %bb7, %bb6
	br label %UnifiedExitNode

bb9:		; preds = %bb5
	%reg107-idxcast3 = sext i32 %fd to i64		; <i64> [#uses=1]
	%idx3 = getelementptr [3 x %spec_fd_t], ptr @spec_fd, i64 0, i64 %reg107-idxcast3, i32 3		; <ptr> [#uses=1]
	%reg1601 = load ptr, ptr %idx3		; <ptr> [#uses=1]
	%reg132-idxcast1 = sext i32 %reg1321 to i64		; <i64> [#uses=1]
	%idx4 = getelementptr i8, ptr %reg1601, i64 %reg132-idxcast1		; <ptr> [#uses=1]
	%reg1621 = load i8, ptr %idx4		; <i8> [#uses=2]
	%cast108 = zext i8 %reg1621 to i64		; <i64> [#uses=0]
	%reg157 = add i32 %reg1321, 1		; <i32> [#uses=1]
	%idx5 = getelementptr [3 x %spec_fd_t], ptr @spec_fd, i64 0, i64 %reg107-idxcast1, i32 2		; <ptr> [#uses=1]
	store i32 %reg157, ptr %idx5
	%reg163 = load i32, ptr @dbglvl		; <i32> [#uses=1]
	%cond272 = icmp sle i32 %reg163, 4		; <i1> [#uses=1]
; CHECK: br label %bb11
	br i1 %cond272, label %bb11, label %bb10

bb10:		; preds = %bb9
	%cast279 = getelementptr [4 x i8], ptr @.LC11, i64 0, i64 0		; <ptr> [#uses=0]
	br label %bb11

bb11:		; preds = %bb10, %bb9
	%cast291 = zext i8 %reg1621 to i32		; <i32> [#uses=1]
	br label %UnifiedExitNode

UnifiedExitNode:		; preds = %bb11, %bb8, %bb4
	%UnifiedRetVal = phi i32 [ 42, %bb4 ], [ -1, %bb8 ], [ %cast291, %bb11 ]		; <i32> [#uses=1]
	ret i32 %UnifiedRetVal
}

declare i32 @puts(ptr)

declare i32 @printf(ptr, ...)
