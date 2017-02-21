; RUN: llc < %s -march=x86 -mcpu=corei7 | FileCheck %s
; RUN: llc < %s -march=x86 -mcpu=corei7-avx | FileCheck %s --check-prefix=SHLD
; RUN: llc < %s -march=x86 -mcpu=core-avx2 | FileCheck %s --check-prefix=BMI2

define i32 @foo(i32 %x, i32 %y, i32 %z) nounwind readnone {
entry:
; CHECK-LABEL: foo:
; CHECK: roll %cl
	%0 = shl i32 %x, %z
	%1 = sub i32 32, %z
	%2 = lshr i32 %x, %1
	%3 = or i32 %2, %0
	ret i32 %3
}

define i32 @bar(i32 %x, i32 %y, i32 %z) nounwind readnone {
entry:
; CHECK-LABEL: bar:
; CHECK: shldl %cl
	%0 = shl i32 %y, %z
	%1 = sub i32 32, %z
	%2 = lshr i32 %x, %1
	%3 = or i32 %2, %0
	ret i32 %3
}

define i32 @un(i32 %x, i32 %y, i32 %z) nounwind readnone {
entry:
; CHECK-LABEL: un:
; CHECK: rorl %cl
	%0 = lshr i32 %x, %z
	%1 = sub i32 32, %z
	%2 = shl i32 %x, %1
	%3 = or i32 %2, %0
	ret i32 %3
}

define i32 @bu(i32 %x, i32 %y, i32 %z) nounwind readnone {
entry:
; CHECK-LABEL: bu:
; CHECK: shrdl %cl
	%0 = lshr i32 %y, %z
	%1 = sub i32 32, %z
	%2 = shl i32 %x, %1
	%3 = or i32 %2, %0
	ret i32 %3
}

define i32 @xfoo(i32 %x, i32 %y, i32 %z) nounwind readnone {
entry:
; CHECK-LABEL: xfoo:
; CHECK: roll $7
; SHLD-LABEL: xfoo:
; SHLD: shldl $7
; BMI2-LABEL: xfoo:
; BMI2: rorxl $25
	%0 = lshr i32 %x, 25
	%1 = shl i32 %x, 7
	%2 = or i32 %0, %1
	ret i32 %2
}

define i32 @xfoop(i32* %p) nounwind readnone {
entry:
; CHECK-LABEL: xfoop:
; CHECK: roll $7
; SHLD-LABEL: xfoop:
; SHLD: shldl $7
; BMI2-LABEL: xfoop:
; BMI2: rorxl $25
	%x = load i32, i32* %p
	%a = lshr i32 %x, 25
	%b = shl i32 %x, 7
	%c = or i32 %a, %b
	ret i32 %c
}

define i32 @xbar(i32 %x, i32 %y, i32 %z) nounwind readnone {
entry:
; CHECK-LABEL: xbar:
; CHECK: shldl $7
	%0 = shl i32 %y, 7
	%1 = lshr i32 %x, 25
	%2 = or i32 %0, %1
	ret i32 %2
}

define i32 @xun(i32 %x, i32 %y, i32 %z) nounwind readnone {
entry:
; CHECK-LABEL: xun:
; CHECK: roll $25
; SHLD-LABEL: xun:
; SHLD: shldl $25
; BMI2-LABEL: xun:
; BMI2: rorxl $7
	%0 = lshr i32 %x, 7
	%1 = shl i32 %x, 25
	%2 = or i32 %0, %1
	ret i32 %2
}

define i32 @xunp(i32* %p) nounwind readnone {
entry:
; CHECK-LABEL: xunp:
; CHECK: roll $25
; shld-label: xunp:
; shld: shldl $25
; BMI2-LABEL: xunp:
; BMI2: rorxl $7
	%x = load i32, i32* %p
	%a = lshr i32 %x, 7
	%b = shl i32 %x, 25
	%c = or i32 %a, %b
	ret i32 %c
}

define i32 @xbu(i32 %x, i32 %y, i32 %z) nounwind readnone {
entry:
; CHECK-LABEL: xbu:
; CHECK: shldl $25
	%0 = lshr i32 %y, 7
	%1 = shl i32 %x, 25
	%2 = or i32 %0, %1
	ret i32 %2
}
