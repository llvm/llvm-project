; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {xor i32 %a, %b} | count 4
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep {and}

define i32 @func1(i32 %a, i32 %b) nounwind readnone {
entry:
	%b_not = xor i32 %b, -1
	%0 = and i32 %a, %b_not
	%a_not = xor i32 %a, -1
	%1 = and i32 %a_not, %b
	%2 = or i32 %0, %1
	ret i32 %2
}

define i32 @func2(i32 %a, i32 %b) nounwind readnone {
entry:
	%b_not = xor i32 %b, -1
	%0 = and i32 %b_not, %a
	%a_not = xor i32 %a, -1
	%1 = and i32 %a_not, %b
	%2 = or i32 %0, %1
	ret i32 %2
}

define i32 @func3(i32 %a, i32 %b) nounwind readnone {
entry:
	%b_not = xor i32 %b, -1
	%0 = and i32 %a, %b_not
	%a_not = xor i32 %a, -1
	%1 = and i32 %b, %a_not
	%2 = or i32 %0, %1
	ret i32 %2
}

define i32 @func4(i32 %a, i32 %b) nounwind readnone {
entry:
	%b_not = xor i32 %b, -1
	%0 = and i32 %b_not, %a
	%a_not = xor i32 %a, -1
	%1 = and i32 %b, %a_not
	%2 = or i32 %0, %1
	ret i32 %2
}
