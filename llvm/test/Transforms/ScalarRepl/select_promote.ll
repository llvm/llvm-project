; Test promotion of loads that use the result of a select instruction.  This
; should be simplified by the instcombine pass.

; RUN: opt < %s -instcombine -mem2reg -S | not grep alloca

define i32 @main() {
	%mem_tmp.0 = alloca i32		; <i32*> [#uses=3]
	%mem_tmp.1 = alloca i32		; <i32*> [#uses=3]
	store i32 0, i32* %mem_tmp.0
	store i32 1, i32* %mem_tmp.1
	%tmp.1.i = load i32* %mem_tmp.1		; <i32> [#uses=1]
	%tmp.3.i = load i32* %mem_tmp.0		; <i32> [#uses=1]
	%tmp.4.i = icmp sle i32 %tmp.1.i, %tmp.3.i		; <i1> [#uses=1]
	%mem_tmp.i.0 = select i1 %tmp.4.i, i32* %mem_tmp.1, i32* %mem_tmp.0		; <i32*> [#uses=1]
	%tmp.3 = load i32* %mem_tmp.i.0		; <i32> [#uses=1]
	ret i32 %tmp.3
}

