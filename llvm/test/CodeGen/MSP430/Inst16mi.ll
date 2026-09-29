; RUN: llc -mtriple=msp430 < %s | FileCheck %s

target datalayout = "e-p:16:8:8-i8:8:8-i16:8:8-i32:8:8"
target triple = "msp430-generic-generic"
@foo = common global i16 0, align 2

define void @mov() nounwind {
; CHECK-LABEL: mov:
; CHECK: mov	#2, &foo
	store i16 2, ptr @foo
	ret void
}

define void @add() nounwind {
; CHECK-LABEL: add:
; CHECK: incd	&foo
	%1 = load i16, ptr @foo
	%2 = add i16 %1, 2
	store i16 %2, ptr @foo
	ret void
}

define void @and() nounwind {
; CHECK-LABEL: and:
; CHECK: and	#2, &foo
	%1 = load i16, ptr @foo
	%2 = and i16 %1, 2
	store i16 %2, ptr @foo
	ret void
}

define void @bis() nounwind {
; CHECK-LABEL: bis:
; CHECK: bis	#2, &foo
	%1 = load i16, ptr @foo
	%2 = or i16 %1, 2
	store i16 %2, ptr @foo
	ret void
}

define void @xor() nounwind {
; CHECK-LABEL: xor:
; CHECK: xor	#2, &foo
	%1 = load i16, ptr @foo
	%2 = xor i16 %1, 2
	store i16 %2, ptr @foo
	ret void
}
