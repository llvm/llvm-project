; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep lshr.*63

define i32 @t1(i64 %d18) {
entry:
	%tmp916 = lshr i64 %d18, 32		; <i64> [#uses=1]
	%tmp917 = trunc i64 %tmp916 to i32		; <i32> [#uses=1]
	%tmp10 = lshr i32 %tmp917, 31		; <i32> [#uses=1]
	ret i32 %tmp10
}

