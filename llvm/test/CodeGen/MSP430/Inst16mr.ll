; RUN: llc -march=msp430 < %s | FileCheck %s
target datalayout = "e-p:16:8:8-i8:8:8-i16:8:8-i32:8:8"
target triple = "msp430-generic-generic"
@foo = common global i16 0, align 2

define void @mov(i16 %a) nounwind {
; CHECK: mov:
; CHECK: mov.w	r15, &foo
	store i16 %a, i16* @foo
	ret void
}

define void @add(i16 %a) nounwind {
; CHECK: add:
; CHECK: add.w	r15, &foo
	%1 = load i16* @foo
	%2 = add i16 %a, %1
	store i16 %2, i16* @foo
	ret void
}

define void @and(i16 %a) nounwind {
; CHECK: and:
; CHECK: and.w	r15, &foo
	%1 = load i16* @foo
	%2 = and i16 %a, %1
	store i16 %2, i16* @foo
	ret void
}

define void @bis(i16 %a) nounwind {
; CHECK: bis:
; CHECK: bis.w	r15, &foo
	%1 = load i16* @foo
	%2 = or i16 %a, %1
	store i16 %2, i16* @foo
	ret void
}

define void @bic(i16 zeroext %m) nounwind {
; CHECK: bic:
; CHECK: bic.w   r15, &foo
        %1 = xor i16 %m, -1
        %2 = load i16* @foo
        %3 = and i16 %2, %1
        store i16 %3, i16* @foo
        ret void
}

define void @xor(i16 %a) nounwind {
; CHECK: xor:
; CHECK: xor.w	r15, &foo
	%1 = load i16* @foo
	%2 = xor i16 %a, %1
	store i16 %2, i16* @foo
	ret void
}

