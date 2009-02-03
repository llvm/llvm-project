; RUN: llvm-as < %s | opt -instcombine

define i65 @foo(i65 %x) nounwind  {
entry:
	%tmp2 = ashr i65 %x, 65		; <i65> [#uses=1]
	ret i65 %tmp2
}
