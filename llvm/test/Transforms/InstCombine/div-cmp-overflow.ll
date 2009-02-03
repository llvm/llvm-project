; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep sdiv
; PR2740

define i1 @func_75(i32 %i2) nounwind {
	%i3 = sdiv i32 %i2, -1328634635
	%i4 = icmp eq i32 %i3, -1
	ret i1 %i4
}
