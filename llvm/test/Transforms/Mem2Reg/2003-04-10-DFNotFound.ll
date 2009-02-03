; RUN: llvm-as < %s | opt -mem2reg

define void @_Z3barv() {
	%result = alloca i32		; <i32*> [#uses=1]
	ret void
		; No predecessors!
	store i32 0, i32* %result
	ret void
}

