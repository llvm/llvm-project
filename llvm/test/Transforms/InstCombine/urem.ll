; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep urem

define i64 @rem_unsigned(i64 %x1, i64 %y2) {
	%r = udiv i64 %x1, %y2
	%r7 = mul i64 %r, %y2
	%r8 = sub i64 %x1, %r7
	ret i64 %r8
}
