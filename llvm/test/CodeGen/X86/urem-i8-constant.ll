; RUN: llc < %s -march=x86 | grep 111

define i8 @foo(i8 %tmp325) {
	%t546 = urem i8 %tmp325, 37
	ret i8 %t546
}
