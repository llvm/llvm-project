; RUN: llc -march=msp430 < %s | FileCheck %s
target datalayout = "e-p:16:8:8-i8:8:8-i16:8:8-i32:8:8"
target triple = "msp430-generic-generic"
@foo = common global i16 0, align 2

define i16 @add(i16 %a) nounwind {
; CHECK: add:
; CHECK: add.w	&foo, r15
	%1 = load i16* @foo
	%2 = add i16 %a, %1
	ret i16 %2
}

define i16 @and(i16 %a) nounwind {
; CHECK: and:
; CHECK: and.w	&foo, r15
	%1 = load i16* @foo
	%2 = and i16 %a, %1
	ret i16 %2
}

define i16 @bis(i16 %a) nounwind {
; CHECK: bis:
; CHECK: bis.w	&foo, r15
	%1 = load i16* @foo
	%2 = or i16 %a, %1
	ret i16 %2
}

define i16  @bic(i16 %a) nounwind {
; CHECK: bic:
; CHECK: bic.w	&foo, r15
        %1 = load i16* @foo
        %2 = xor i16 %1, -1
        %3 = and i16 %a, %2
        ret i16 %3
}

define i16 @xor(i16 %a) nounwind {
; CHECK: xor:
; CHECK: xor.w	&foo, r15
	%1 = load i16* @foo
	%2 = xor i16 %a, %1
	ret i16 %2
}

