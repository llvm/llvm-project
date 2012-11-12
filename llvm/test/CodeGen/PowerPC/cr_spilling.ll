; RUN: llc < %s -march=ppc32 -regalloc=fast -O0 -relocation-model=pic -o -
; PR1638

@.str242 = external constant [3 x i8]		; <[3 x i8]*> [#uses=1]

define fastcc void @ParseContent(i8* %buf, i32 %bufsize) {
entry:
	%items = alloca [10000 x i8*], align 16		; <[10000 x i8*]*> [#uses=0]
	%tmp86 = add i32 0, -1		; <i32> [#uses=1]
	br i1 false, label %cond_true94, label %cond_next99

cond_true94:		; preds = %entry
	%tmp98 = call i32 (i8*, ...)* @printf(i8* getelementptr ([3 x i8]* @.str242, i32 0, i32 0), i8* null)		; <i32> [#uses=0]
	%tmp20971 = icmp sgt i32 %tmp86, 0		; <i1> [#uses=1]
	br i1 %tmp20971, label %bb101, label %bb212

cond_next99:		; preds = %entry
	ret void

bb101:		; preds = %cond_true94
	ret void

bb212:		; preds = %cond_true94
	ret void
}

declare i32 @printf(i8*, ...)
