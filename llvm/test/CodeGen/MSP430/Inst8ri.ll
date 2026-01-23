; RUN: llc -mtriple=msp430 < %s | FileCheck %s
target datalayout = "e-p:16:8:8-i8:8:8-i16:8:8-i32:8:8"
target triple = "msp430-generic-generic"

define i8 @mov() nounwind {
; CHECK-LABEL: mov:
; CHECK: mov.b	#1, r12
	ret i8 1
}

define i8 @add(i8 %a, i8 %b) nounwind {
; CHECK-LABEL: add:
; CHECK: inc.b	r12
	%1 = add i8 %a, 1
	ret i8 %1
}

define i8 @and(i8 %a, i8 %b) nounwind {
; CHECK-LABEL: and:
; CHECK: and.b	#1, r12
	%1 = and i8 %a, 1
	ret i8 %1
}

define i8 @bis(i8 %a, i8 %b) nounwind {
; CHECK-LABEL: bis:
; CHECK: bis.b	#1, r12
	%1 = or i8 %a, 1
	ret i8 %1
}

define i8 @xor(i8 %a, i8 %b) nounwind {
; CHECK-LABEL: xor:
; CHECK: xor.b	#1, r12
	%1 = xor i8 %a, 1
	ret i8 %1
}
