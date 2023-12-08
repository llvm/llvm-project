; RUN: llc -march=msp430 < %s | FileCheck %s
target datalayout = "e-p:16:8:8-i8:8:8-i16:8:8-i32:8:8"
target triple = "msp430-generic-generic"
@foo = common global i16 0, align 2

define i16 @add(i16 %a) nounwind {
; CHECK-LABEL: add:
; CHECK: add	&foo, r12
	%1 = load i16, i16* @foo
	%2 = add i16 %a, %1
	ret i16 %2
}

define i16 @and(i16 %a) nounwind {
; CHECK-LABEL: and:
; CHECK: and	&foo, r12
	%1 = load i16, i16* @foo
	%2 = and i16 %a, %1
	ret i16 %2
}

define i16 @bis(i16 %a) nounwind {
; CHECK-LABEL: bis:
; CHECK: bis	&foo, r12
	%1 = load i16, i16* @foo
	%2 = or i16 %a, %1
	ret i16 %2
}

define i16  @bic(i16 %a) nounwind {
; CHECK-LABEL: bic:
; CHECK: bic	&foo, r12
        %1 = load i16, i16* @foo
        %2 = xor i16 %1, -1
        %3 = and i16 %a, %2
        ret i16 %3
}

define i16 @xor(i16 %a) nounwind {
; CHECK-LABEL: xor:
; CHECK: xor	&foo, r12
	%1 = load i16, i16* @foo
	%2 = xor i16 %a, %1
	ret i16 %2
}

