; RUN: llc < %s -march=x86 > %t
; RUN: grep movsbl %t
; RUN: not grep movz %t
; RUN: not grep and %t

define signext  i8 @foo(i16 signext  %x) nounwind  {
	%retval56 = trunc i16 %x to i8
	ret i8 %retval56
}
