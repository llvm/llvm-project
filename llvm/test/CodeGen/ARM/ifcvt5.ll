; RUN: llvm-as < %s | llc -march=arm
; RUN: llvm-as < %s | llc -march=arm | grep blge | count 1

@x = external global i32*		; <i32**> [#uses=1]

define void @foo(i32 %a) {
entry:
	%tmp = load i32** @x		; <i32*> [#uses=1]
	store i32 %a, i32* %tmp
	ret void
}

define void @t1(i32 %a, i32 %b) {
entry:
	%tmp1 = icmp sgt i32 %a, 10		; <i1> [#uses=1]
	br i1 %tmp1, label %cond_true, label %UnifiedReturnBlock

cond_true:		; preds = %entry
	tail call void @foo( i32 %b )
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}
