; RUN: llc < %s -mtriple=ppc32-- -regalloc=fast -O0 -relocation-model=pic -o -
; PR1638

@.str242 = external constant [3 x i8]		; <ptr> [#uses=1]

define fastcc void @ParseContent(ptr %buf, i32 %bufsize) {
entry:
	%items = alloca [10000 x ptr], align 16		; <ptr> [#uses=0]
	%tmp86 = add i32 0, -1		; <i32> [#uses=1]
	br i1 false, label %cond_true94, label %cond_next99

cond_true94:		; preds = %entry
	%tmp98 = call i32 (ptr, ...) @printf(ptr @.str242, ptr null)		; <i32> [#uses=0]
	%tmp20971 = icmp sgt i32 %tmp86, 0		; <i1> [#uses=1]
	br i1 %tmp20971, label %bb101, label %bb212

cond_next99:		; preds = %entry
	ret void

bb101:		; preds = %cond_true94
	ret void

bb212:		; preds = %cond_true94
	ret void
}

declare i32 @printf(ptr, ...)
