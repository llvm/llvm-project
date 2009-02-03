; RUN: llvm-as < %s | opt -globalopt | llvm-dis
; PR1896

@indirect1 = internal global void (i32)* null		; <void (i32)**> [#uses=2]

declare void @indirectmarked(i32)

define i32 @main() {
entry:
	br i1 false, label %cond_next20.i, label %cond_true.i9

cond_true.i9:		; preds = %entry
	ret i32 0

cond_next20.i:		; preds = %entry
	store void (i32)* @indirectmarked, void (i32)** @indirect1, align 4
	br i1 false, label %cond_next21.i.i23.i, label %stack_restore

stack_restore:		; preds = %cond_next20.i
	ret i32 0

cond_next21.i.i23.i:		; preds = %cond_next20.i
	%tmp6.i4.i = load i32* bitcast (void (i32)** @indirect1 to i32*), align 4		; <i32> [#uses=0]
	ret i32 0
}

