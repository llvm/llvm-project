; RUN: llvm-as < %s | opt -globalopt | llvm-dis | grep {tmp.f1 = phi i32. }
; RUN: llvm-as < %s | opt -globalopt | llvm-dis | grep {tmp.f0 = phi i32. }

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin7"
	%struct.foo = type { i32, i32 }
@X = internal global %struct.foo* null		; <%struct.foo**> [#uses=2]

define void @bar(i32 %Size) nounwind noinline {
entry:
	%tmp = malloc [1000000 x %struct.foo]		; <[1000000 x %struct.foo]*> [#uses=1]
	%.sub = getelementptr [1000000 x %struct.foo]* %tmp, i32 0, i32 0		; <%struct.foo*> [#uses=1]
	store %struct.foo* %.sub, %struct.foo** @X, align 4
	ret void
}

define i32 @baz() nounwind readonly noinline {
bb1.thread:
	%tmpLD1 = load %struct.foo** @X, align 4		; <%struct.foo*> [#uses=1]
	br label %bb1

bb1:		; preds = %bb1, %bb1.thread
        %tmp = phi %struct.foo* [%tmpLD1, %bb1.thread ], [ %tmpLD2, %bb1 ]		; <i32> [#uses=2]
	%i.0.reg2mem.0 = phi i32 [ 0, %bb1.thread ], [ %indvar.next, %bb1 ]		; <i32> [#uses=2]
	%sum.0.reg2mem.0 = phi i32 [ 0, %bb1.thread ], [ %tmp3, %bb1 ]		; <i32> [#uses=1]
	%tmp1 = getelementptr %struct.foo* %tmp, i32 %i.0.reg2mem.0, i32 0		; <i32*> [#uses=1]
	%tmp2 = load i32* %tmp1, align 4		; <i32> [#uses=1]
	%tmp6 = add i32 %tmp2, %sum.0.reg2mem.0		; <i32> [#uses=2]
	%tmp4 = getelementptr %struct.foo* %tmp, i32 %i.0.reg2mem.0, i32 1		; <i32*> [#uses=1]
        %tmp5 = load i32 * %tmp4
        %tmp3 = add i32 %tmp5, %tmp6
	%indvar.next = add i32 %i.0.reg2mem.0, 1		; <i32> [#uses=2]
        
      	%tmpLD2 = load %struct.foo** @X, align 4		; <%struct.foo*> [#uses=1]

	%exitcond = icmp eq i32 %indvar.next, 1200		; <i1> [#uses=1]
	br i1 %exitcond, label %bb2, label %bb1

bb2:		; preds = %bb1
	ret i32 %tmp3
}
