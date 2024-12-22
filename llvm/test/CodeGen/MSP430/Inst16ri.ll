; RUN: llc -march=msp430 < %s | FileCheck %s
target datalayout = "e-p:16:8:8-i8:8:8-i16:8:8-i32:8:8"
target triple = "msp430-generic-generic"

define i16 @mov() nounwind {
; CHECK-LABEL: mov:
; CHECK: mov	#1, r12
	ret i16 1
}

define i16 @add(i16 %a, i16 %b) nounwind {
; CHECK-LABEL: add:
; CHECK: inc	r12
	%1 = add i16 %a, 1
	ret i16 %1
}

define i16 @and(i16 %a, i16 %b) nounwind {
; CHECK-LABEL: and:
; CHECK: and	#1, r12
	%1 = and i16 %a, 1
	ret i16 %1
}

define i16 @bis(i16 %a, i16 %b) nounwind {
; CHECK-LABEL: bis:
; CHECK: bis	#1, r12
	%1 = or i16 %a, 1
	ret i16 %1
}

define i16 @xor(i16 %a, i16 %b) nounwind {
; CHECK-LABEL: xor:
; CHECK: xor	#1, r12
	%1 = xor i16 %a, 1
	ret i16 %1
}
