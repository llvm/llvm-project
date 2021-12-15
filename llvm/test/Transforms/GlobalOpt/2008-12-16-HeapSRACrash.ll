; RUN: opt < %s -passes=globalopt | llvm-dis
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin7"
	%struct.foo = type { i32, i32 }
@X = internal global %struct.foo* null		; <%struct.foo**> [#uses=2]

define void @bar(i32 %Size) nounwind noinline {
entry:
        %malloccall = tail call i8* @malloc(i32 trunc (i64 mul (i64 ptrtoint (i32* getelementptr (i32, i32* null, i32 1) to i64), i64 2000000) to i32))
        %tmp = bitcast i8* %malloccall to [1000000 x %struct.foo]*
	%.sub = getelementptr [1000000 x %struct.foo], [1000000 x %struct.foo]* %tmp, i32 0, i32 0		; <%struct.foo*> [#uses=1]
	store %struct.foo* %.sub, %struct.foo** @X, align 4
	ret void
}

declare noalias i8* @malloc(i32)

define i32 @baz() nounwind readonly noinline {
bb1.thread:
	%tmpLD1 = load %struct.foo*, %struct.foo** @X, align 4		; <%struct.foo*> [#uses=3]
	store %struct.foo* %tmpLD1, %struct.foo** null
	br label %bb1

bb1:		; preds = %bb1, %bb1.thread
	%tmp = phi %struct.foo* [ %tmpLD1, %bb1.thread ], [ %tmpLD1, %bb1 ]		; <%struct.foo*> [#uses=0]
	br i1 false, label %bb2, label %bb1

bb2:		; preds = %bb1
	ret i32 0
}
