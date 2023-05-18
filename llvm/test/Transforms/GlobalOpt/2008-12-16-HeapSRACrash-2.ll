; RUN: opt < %s -passes=globalopt | llvm-dis
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin7"
	%struct.foo = type { i32, i32 }
@X = internal global ptr null		; <ptr> [#uses=2]

define void @bar(i32 %Size) nounwind noinline {
entry:
        %malloccall = tail call ptr @malloc(i32 trunc (i64 mul (i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 1) to i64), i64 2000000) to i32))
	%.sub = getelementptr [1000000 x %struct.foo], ptr %malloccall, i32 0, i32 0		; <ptr> [#uses=1]
	store ptr %.sub, ptr @X, align 4
	ret void
}

declare noalias ptr @malloc(i32)


define i32 @baz() nounwind readonly noinline {
bb1.thread:
	%tmpLD1 = load ptr, ptr @X, align 4		; <ptr> [#uses=2]
	br label %bb1

bb1:		; preds = %bb1, %bb1.thread
	%tmp = phi ptr [ %tmpLD1, %bb1.thread ], [ %tmpLD1, %bb1 ]		; <ptr> [#uses=1]
	%0 = getelementptr %struct.foo, ptr %tmp, i32 1		; <ptr> [#uses=0]
	br label %bb1
}
