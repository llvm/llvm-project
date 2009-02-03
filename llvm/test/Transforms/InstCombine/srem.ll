; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep srem

define i64 @foo(i64 %x1, i64 %y2) {
	%r = sdiv i64 %x1, %y2
	%r7 = mul i64 %r, %y2
	%r8 = sub i64 %x1, %r7
	ret i64 %r8
}
