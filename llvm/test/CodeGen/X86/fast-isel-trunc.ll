; RUN: llvm-as < %s | llc -march=x86 -fast-isel
; RUN: llvm-as < %s | llc -march=x86-64 -fast-isel

define i8 @t1(i32 %x) signext nounwind  {
	%tmp1 = trunc i32 %x to i8
	ret i8 %tmp1
}

define i8 @t2(i16 signext %x) signext nounwind  {
	%tmp1 = trunc i16 %x to i8
	ret i8 %tmp1
}
