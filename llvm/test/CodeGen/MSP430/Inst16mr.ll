; RUN: llc -march=msp430 < %s | FileCheck %s
target datalayout = "e-p:16:8:8-i8:8:8-i16:8:8-i32:8:8"
target triple = "msp430-generic-generic"
@foo = common global i16 0, align 2

define void @mov(i16 %a) nounwind {
; CHECK-LABEL: mov:
; CHECK: mov.w	r12, &foo
	store i16 %a, i16* @foo
	ret void
}

define void @add(i16 %a) nounwind {
; CHECK-LABEL: add:
; CHECK: add.w	r12, &foo
	%1 = load i16, i16* @foo
	%2 = add i16 %a, %1
	store i16 %2, i16* @foo
	ret void
}

define void @and(i16 %a) nounwind {
; CHECK-LABEL: and:
; CHECK: and.w	r12, &foo
	%1 = load i16, i16* @foo
	%2 = and i16 %a, %1
	store i16 %2, i16* @foo
	ret void
}

define void @bis(i16 %a) nounwind {
; CHECK-LABEL: bis:
; CHECK: bis.w	r12, &foo
	%1 = load i16, i16* @foo
	%2 = or i16 %a, %1
	store i16 %2, i16* @foo
	ret void
}

define void @bic(i16 zeroext %m) nounwind {
; CHECK-LABEL: bic:
; CHECK: bic.w   r12, &foo
        %1 = xor i16 %m, -1
        %2 = load i16, i16* @foo
        %3 = and i16 %2, %1
        store i16 %3, i16* @foo
        ret void
}

define void @xor(i16 %a) nounwind {
; CHECK-LABEL: xor:
; CHECK: xor.w	r12, &foo
	%1 = load i16, i16* @foo
	%2 = xor i16 %a, %1
	store i16 %2, i16* @foo
	ret void
}

