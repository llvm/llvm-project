; RUN: llc < %s -march=x86 | FileCheck %s

define i16 @foo(i16 %x, i16 %y, i16 %z) nounwind readnone {
entry:
; CHECK: foo:
; CHECK: rolw %cl
	%0 = shl i16 %x, %z
	%1 = sub i16 16, %z
	%2 = lshr i16 %x, %1
	%3 = or i16 %2, %0
	ret i16 %3
}

define i16 @bar(i16 %x, i16 %y, i16 %z) nounwind readnone {
entry:
; CHECK: bar:
; CHECK: shldw %cl
	%0 = shl i16 %y, %z
	%1 = sub i16 16, %z
	%2 = lshr i16 %x, %1
	%3 = or i16 %2, %0
	ret i16 %3
}

define i16 @un(i16 %x, i16 %y, i16 %z) nounwind readnone {
entry:
; CHECK: un:
; CHECK: rorw %cl
	%0 = lshr i16 %x, %z
	%1 = sub i16 16, %z
	%2 = shl i16 %x, %1
	%3 = or i16 %2, %0
	ret i16 %3
}

define i16 @bu(i16 %x, i16 %y, i16 %z) nounwind readnone {
entry:
; CHECK: bu:
; CHECK: shrdw
	%0 = lshr i16 %y, %z
	%1 = sub i16 16, %z
	%2 = shl i16 %x, %1
	%3 = or i16 %2, %0
	ret i16 %3
}

define i16 @xfoo(i16 %x, i16 %y, i16 %z) nounwind readnone {
entry:
; CHECK: xfoo:
; CHECK: rolw $5
	%0 = lshr i16 %x, 11
	%1 = shl i16 %x, 5
	%2 = or i16 %0, %1
	ret i16 %2
}

define i16 @xbar(i16 %x, i16 %y, i16 %z) nounwind readnone {
entry:
; CHECK: xbar:
; CHECK: shldw $5
	%0 = shl i16 %y, 5
	%1 = lshr i16 %x, 11
	%2 = or i16 %0, %1
	ret i16 %2
}

define i16 @xun(i16 %x, i16 %y, i16 %z) nounwind readnone {
entry:
; CHECK: xun:
; CHECK: rolw $11
	%0 = lshr i16 %x, 5
	%1 = shl i16 %x, 11
	%2 = or i16 %0, %1
	ret i16 %2
}

define i16 @xbu(i16 %x, i16 %y, i16 %z) nounwind readnone {
entry:
; CHECK: xbu:
; CHECK: shldw $11
	%0 = lshr i16 %y, 5
	%1 = shl i16 %x, 11
	%2 = or i16 %0, %1
	ret i16 %2
}
