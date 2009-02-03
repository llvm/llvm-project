; RUN: llvm-as < %s | llc -march=x86 > %t
; RUN: grep rol %t | count 3
; RUN: grep ror %t | count 1
; RUN: grep shld %t | count 2
; RUN: grep shrd %t | count 2

define i16 @foo(i16 %x, i16 %y, i16 %z) nounwind readnone {
entry:
	%0 = shl i16 %x, %z
	%1 = sub i16 16, %z
	%2 = lshr i16 %x, %1
	%3 = or i16 %2, %0
	ret i16 %3
}

define i16 @bar(i16 %x, i16 %y, i16 %z) nounwind readnone {
entry:
	%0 = shl i16 %y, %z
	%1 = sub i16 16, %z
	%2 = lshr i16 %x, %1
	%3 = or i16 %2, %0
	ret i16 %3
}

define i16 @un(i16 %x, i16 %y, i16 %z) nounwind readnone {
entry:
	%0 = lshr i16 %x, %z
	%1 = sub i16 16, %z
	%2 = shl i16 %x, %1
	%3 = or i16 %2, %0
	ret i16 %3
}

define i16 @bu(i16 %x, i16 %y, i16 %z) nounwind readnone {
entry:
	%0 = lshr i16 %y, %z
	%1 = sub i16 16, %z
	%2 = shl i16 %x, %1
	%3 = or i16 %2, %0
	ret i16 %3
}

define i16 @xfoo(i16 %x, i16 %y, i16 %z) nounwind readnone {
entry:
	%0 = lshr i16 %x, 11
	%1 = shl i16 %x, 5
	%2 = or i16 %0, %1
	ret i16 %2
}

define i16 @xbar(i16 %x, i16 %y, i16 %z) nounwind readnone {
entry:
	%0 = shl i16 %y, 5
	%1 = lshr i16 %x, 11
	%2 = or i16 %0, %1
	ret i16 %2
}

define i16 @xun(i16 %x, i16 %y, i16 %z) nounwind readnone {
entry:
	%0 = lshr i16 %x, 5
	%1 = shl i16 %x, 11
	%2 = or i16 %0, %1
	ret i16 %2
}

define i16 @xbu(i16 %x, i16 %y, i16 %z) nounwind readnone {
entry:
	%0 = lshr i16 %y, 5
	%1 = shl i16 %x, 11
	%2 = or i16 %0, %1
	ret i16 %2
}
