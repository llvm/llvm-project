; RUN: llvm-as < %s | opt -globalopt | llvm-dis | grep {@X.f0}
; RUN: llvm-as < %s | opt -globalopt | llvm-dis | grep {@X.f1}

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin7"
	%struct.foo = type { i32, i32 }
@X = internal global %struct.foo* null		; <%struct.foo**> [#uses=2]

define void @bar(i32 %Size) nounwind noinline {
entry:
	%0 = malloc [1000000 x %struct.foo]
        ;%.sub = bitcast [1000000 x %struct.foo]* %0 to %struct.foo*
	%.sub = getelementptr [1000000 x %struct.foo]* %0, i32 0, i32 0		; <%struct.foo*> [#uses=1]
	store %struct.foo* %.sub, %struct.foo** @X, align 4
	ret void
}

define i32 @baz() nounwind readonly noinline {
bb1.thread:
	%0 = load %struct.foo** @X, align 4		; <%struct.foo*> [#uses=1]
	br label %bb1

bb1:		; preds = %bb1, %bb1.thread
	%i.0.reg2mem.0 = phi i32 [ 0, %bb1.thread ], [ %indvar.next, %bb1 ]		; <i32> [#uses=2]
	%sum.0.reg2mem.0 = phi i32 [ 0, %bb1.thread ], [ %3, %bb1 ]		; <i32> [#uses=1]
	%1 = getelementptr %struct.foo* %0, i32 %i.0.reg2mem.0, i32 0		; <i32*> [#uses=1]
	%2 = load i32* %1, align 4		; <i32> [#uses=1]
	%3 = add i32 %2, %sum.0.reg2mem.0		; <i32> [#uses=2]
	%indvar.next = add i32 %i.0.reg2mem.0, 1		; <i32> [#uses=2]
	%exitcond = icmp eq i32 %indvar.next, 1200		; <i1> [#uses=1]
	br i1 %exitcond, label %bb2, label %bb1

bb2:		; preds = %bb1
	ret i32 %3
}

