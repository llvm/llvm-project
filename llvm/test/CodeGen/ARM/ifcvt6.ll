; RUN: llvm-as < %s | \
; RUN:   llc -march=arm -mtriple=arm-apple-darwin
; RUN: llvm-as < %s | \
; RUN:   llc -march=arm -mtriple=arm-apple-darwin | \
; RUN:   grep cmpne | count 1
; RUN: llvm-as < %s | \
; RUN:   llc -march=arm -mtriple=arm-apple-darwin | \
; RUN:   grep ldmhi | count 1

define void @foo(i32 %X, i32 %Y) {
entry:
	%tmp1 = icmp ult i32 %X, 4		; <i1> [#uses=1]
	%tmp4 = icmp eq i32 %Y, 0		; <i1> [#uses=1]
	%tmp7 = or i1 %tmp4, %tmp1		; <i1> [#uses=1]
	br i1 %tmp7, label %cond_true, label %UnifiedReturnBlock

cond_true:		; preds = %entry
	%tmp10 = tail call i32 (...)* @bar( )		; <i32> [#uses=0]
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

declare i32 @bar(...)
