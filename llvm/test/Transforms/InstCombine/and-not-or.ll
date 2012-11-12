; RUN: opt < %s -instcombine -S | grep "and i32 %x, %y" | count 4
; RUN: opt < %s -instcombine -S | not grep "or"

define i32 @func1(i32 %x, i32 %y) nounwind {
entry:
	%n = xor i32 %y, -1
	%o = or i32 %n, %x
	%a = and i32 %o, %y
	ret i32 %a
}

define i32 @func2(i32 %x, i32 %y) nounwind {
entry:
	%n = xor i32 %y, -1
	%o = or i32 %x, %n
	%a = and i32 %o, %y
	ret i32 %a
}

define i32 @func3(i32 %x, i32 %y) nounwind {
entry:
	%n = xor i32 %y, -1
	%o = or i32 %n, %x
	%a = and i32 %y, %o
	ret i32 %a
}

define i32 @func4(i32 %x, i32 %y) nounwind {
entry:
	%n = xor i32 %y, -1
	%o = or i32 %x, %n
	%a = and i32 %y, %o
	ret i32 %a
}
