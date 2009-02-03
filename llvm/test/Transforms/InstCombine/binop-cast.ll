; RUN: llvm-as < %s | opt -instcombine | llvm-dis | notcast

define i32 @testAdd(i32 %X, i32 %Y) {
	%tmp = add i32 %X, %Y
	%tmp.l = bitcast i32 %tmp to i32
	ret i32 %tmp.l
}
