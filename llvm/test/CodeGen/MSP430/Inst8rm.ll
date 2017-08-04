; RUN: llc -march=msp430 < %s | FileCheck %s
target datalayout = "e-p:16:8:8-i8:8:8-i8:8:8-i32:8:8"
target triple = "msp430-generic-generic"
@foo = common global i8 0, align 1

define i8 @add(i8 %a) nounwind {
; CHECK-LABEL: add:
; CHECK: add.b	&foo, r12
	%1 = load i8, i8* @foo
	%2 = add i8 %a, %1
	ret i8 %2
}

define i8 @and(i8 %a) nounwind {
; CHECK-LABEL: and:
; CHECK: and.b	&foo, r12
	%1 = load i8, i8* @foo
	%2 = and i8 %a, %1
	ret i8 %2
}

define i8 @bis(i8 %a) nounwind {
; CHECK-LABEL: bis:
; CHECK: bis.b	&foo, r12
	%1 = load i8, i8* @foo
	%2 = or i8 %a, %1
	ret i8 %2
}

define i8  @bic(i8 %a) nounwind {
; CHECK-LABEL: bic:
; CHECK: bic.b  &foo, r12
        %1 = load i8, i8* @foo
        %2 = xor i8 %1, -1
        %3 = and i8 %a, %2
        ret i8 %3
}

define i8 @xor(i8 %a) nounwind {
; CHECK-LABEL: xor:
; CHECK: xor.b	&foo, r12
	%1 = load i8, i8* @foo
	%2 = xor i8 %a, %1
	ret i8 %2
}

