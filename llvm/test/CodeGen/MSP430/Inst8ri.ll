; RUN: llc -march=msp430 < %s | FileCheck %s
target datalayout = "e-p:16:8:8-i8:8:8-i16:8:8-i32:8:8"
target triple = "msp430-generic-generic"

define i8 @mov() nounwind {
; CHECK: mov:
; CHECK: mov.b	#1, r15
	ret i8 1
}

define i8 @add(i8 %a, i8 %b) nounwind {
; CHECK: add:
; CHECK: add.b	#1, r15
	%1 = add i8 %a, 1
	ret i8 %1
}

define i8 @and(i8 %a, i8 %b) nounwind {
; CHECK: and:
; CHECK: and.b	#1, r15
	%1 = and i8 %a, 1
	ret i8 %1
}

define i8 @bis(i8 %a, i8 %b) nounwind {
; CHECK: bis:
; CHECK: bis.b	#1, r15
	%1 = or i8 %a, 1
	ret i8 %1
}

define i8 @xor(i8 %a, i8 %b) nounwind {
; CHECK: xor:
; CHECK: xor.b	#1, r15
	%1 = xor i8 %a, 1
	ret i8 %1
}
