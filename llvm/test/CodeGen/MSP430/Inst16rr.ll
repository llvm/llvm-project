; RUN: llc -march=msp430 < %s | FileCheck %s
target datalayout = "e-p:16:8:8-i8:8:8-i16:8:8-i32:8:8"
target triple = "msp430-generic-generic"

define i16 @mov(i16 %a, i16 %b) nounwind {
; CHECK: mov:
; CHECK: mov.w	r14, r15
	ret i16 %b
}

define i16 @add(i16 %a, i16 %b) nounwind {
; CHECK: add:
; CHECK: add.w	r14, r15
	%1 = add i16 %a, %b
	ret i16 %1
}

define i16 @and(i16 %a, i16 %b) nounwind {
; CHECK: and:
; CHECK: and.w	r14, r15
	%1 = and i16 %a, %b
	ret i16 %1
}

define i16 @bis(i16 %a, i16 %b) nounwind {
; CHECK: bis:
; CHECK: bis.w	r14, r15
	%1 = or i16 %a, %b
	ret i16 %1
}

define i16 @bic(i16 %a, i16 %b) nounwind {
; CHECK: bic:
; CHECK: bic.w	r14, r15
        %1 = xor i16 %b, -1
        %2 = and i16 %a, %1
        ret i16 %2
}

define i16 @xor(i16 %a, i16 %b) nounwind {
; CHECK: xor:
; CHECK: xor.w	r14, r15
	%1 = xor i16 %a, %b
	ret i16 %1
}
