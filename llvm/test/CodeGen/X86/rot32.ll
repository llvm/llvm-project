; RUN: llvm-as < %s | llc -march=x86 > %t
; RUN: grep rol %t | count 3
; RUN: grep ror %t | count 1
; RUN: grep shld %t | count 2
; RUN: grep shrd %t | count 2

define i32 @foo(i32 %x, i32 %y, i32 %z) nounwind readnone {
entry:
	%0 = shl i32 %x, %z
	%1 = sub i32 32, %z
	%2 = lshr i32 %x, %1
	%3 = or i32 %2, %0
	ret i32 %3
}

define i32 @bar(i32 %x, i32 %y, i32 %z) nounwind readnone {
entry:
	%0 = shl i32 %y, %z
	%1 = sub i32 32, %z
	%2 = lshr i32 %x, %1
	%3 = or i32 %2, %0
	ret i32 %3
}

define i32 @un(i32 %x, i32 %y, i32 %z) nounwind readnone {
entry:
	%0 = lshr i32 %x, %z
	%1 = sub i32 32, %z
	%2 = shl i32 %x, %1
	%3 = or i32 %2, %0
	ret i32 %3
}

define i32 @bu(i32 %x, i32 %y, i32 %z) nounwind readnone {
entry:
	%0 = lshr i32 %y, %z
	%1 = sub i32 32, %z
	%2 = shl i32 %x, %1
	%3 = or i32 %2, %0
	ret i32 %3
}

define i32 @xfoo(i32 %x, i32 %y, i32 %z) nounwind readnone {
entry:
	%0 = lshr i32 %x, 25
	%1 = shl i32 %x, 7
	%2 = or i32 %0, %1
	ret i32 %2
}

define i32 @xbar(i32 %x, i32 %y, i32 %z) nounwind readnone {
entry:
	%0 = shl i32 %y, 7
	%1 = lshr i32 %x, 25
	%2 = or i32 %0, %1
	ret i32 %2
}

define i32 @xun(i32 %x, i32 %y, i32 %z) nounwind readnone {
entry:
	%0 = lshr i32 %x, 7
	%1 = shl i32 %x, 25
	%2 = or i32 %0, %1
	ret i32 %2
}

define i32 @xbu(i32 %x, i32 %y, i32 %z) nounwind readnone {
entry:
	%0 = lshr i32 %y, 7
	%1 = shl i32 %x, 25
	%2 = or i32 %0, %1
	ret i32 %2
}
