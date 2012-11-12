; RUN: llc < %s -march=x86-64 -mcpu=corei7 > %t
; RUN: grep rol %t | count 5
; RUN: grep ror %t | count 1
; RUN: grep shld %t | count 2
; RUN: grep shrd %t | count 2
; RUN: llc < %s -march=x86-64 -mcpu=core-avx2 | FileCheck %s --check-prefix=BMI2

define i64 @foo(i64 %x, i64 %y, i64 %z) nounwind readnone {
entry:
	%0 = shl i64 %x, %z
	%1 = sub i64 64, %z
	%2 = lshr i64 %x, %1
	%3 = or i64 %2, %0
	ret i64 %3
}

define i64 @bar(i64 %x, i64 %y, i64 %z) nounwind readnone {
entry:
	%0 = shl i64 %y, %z
	%1 = sub i64 64, %z
	%2 = lshr i64 %x, %1
	%3 = or i64 %2, %0
	ret i64 %3
}

define i64 @un(i64 %x, i64 %y, i64 %z) nounwind readnone {
entry:
	%0 = lshr i64 %x, %z
	%1 = sub i64 64, %z
	%2 = shl i64 %x, %1
	%3 = or i64 %2, %0
	ret i64 %3
}

define i64 @bu(i64 %x, i64 %y, i64 %z) nounwind readnone {
entry:
	%0 = lshr i64 %y, %z
	%1 = sub i64 64, %z
	%2 = shl i64 %x, %1
	%3 = or i64 %2, %0
	ret i64 %3
}

define i64 @xfoo(i64 %x, i64 %y, i64 %z) nounwind readnone {
entry:
; BMI2: xfoo:
; BMI2: rorxq $57
	%0 = lshr i64 %x, 57
	%1 = shl i64 %x, 7
	%2 = or i64 %0, %1
	ret i64 %2
}

define i64 @xfoop(i64* %p) nounwind readnone {
entry:
; BMI2: xfoop:
; BMI2: rorxq $57, ({{.+}}), %{{.+}}
	%x = load i64* %p
	%a = lshr i64 %x, 57
	%b = shl i64 %x, 7
	%c = or i64 %a, %b
	ret i64 %c
}

define i64 @xbar(i64 %x, i64 %y, i64 %z) nounwind readnone {
entry:
	%0 = shl i64 %y, 7
	%1 = lshr i64 %x, 57
	%2 = or i64 %0, %1
	ret i64 %2
}

define i64 @xun(i64 %x, i64 %y, i64 %z) nounwind readnone {
entry:
; BMI2: xun:
; BMI2: rorxq $7
	%0 = lshr i64 %x, 7
	%1 = shl i64 %x, 57
	%2 = or i64 %0, %1
	ret i64 %2
}

define i64 @xunp(i64* %p) nounwind readnone {
entry:
; BMI2: xunp:
; BMI2: rorxq $7, ({{.+}}), %{{.+}}
	%x = load i64* %p
	%a = lshr i64 %x, 7
	%b = shl i64 %x, 57
	%c = or i64 %a, %b
	ret i64 %c
}

define i64 @xbu(i64 %x, i64 %y, i64 %z) nounwind readnone {
entry:
	%0 = lshr i64 %y, 7
	%1 = shl i64 %x, 57
	%2 = or i64 %0, %1
	ret i64 %2
}
