; RUN: llvm-as < %s | llc -march=x86 | grep movsbl

define i32 @foo(i32 %X) nounwind  {
entry:
	%tmp12 = trunc i32 %X to i8		; <i8> [#uses=1]
	%tmp123 = sext i8 %tmp12 to i32		; <i32> [#uses=1]
	ret i32 %tmp123
}

