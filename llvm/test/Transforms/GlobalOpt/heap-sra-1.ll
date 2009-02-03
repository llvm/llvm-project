; RUN: llvm-as < %s | opt -globalopt | llvm-dis | grep {@X.f0}
; RUN: llvm-as < %s | opt -globalopt | llvm-dis | grep {@X.f1}
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin7"

	%struct.foo = type { i32, i32 }
@X = internal global %struct.foo* null

define void @bar(i32 %Size) nounwind noinline {
entry:
	%.sub = malloc %struct.foo, i32 %Size	
	store %struct.foo* %.sub, %struct.foo** @X, align 4
	ret void
}

define i32 @baz() nounwind readonly noinline {
bb1.thread:
	%0 = load %struct.foo** @X, align 4		
	br label %bb1

bb1:		; preds = %bb1, %bb1.thread
	%i.0.reg2mem.0 = phi i32 [ 0, %bb1.thread ], [ %indvar.next, %bb1 ]
	%sum.0.reg2mem.0 = phi i32 [ 0, %bb1.thread ], [ %3, %bb1 ]
	%1 = getelementptr %struct.foo* %0, i32 %i.0.reg2mem.0, i32 0
	%2 = load i32* %1, align 4
	%3 = add i32 %2, %sum.0.reg2mem.0	
	%indvar.next = add i32 %i.0.reg2mem.0, 1	
	%exitcond = icmp eq i32 %indvar.next, 1200		
	br i1 %exitcond, label %bb2, label %bb1

bb2:		; preds = %bb1
	ret i32 %3
}

