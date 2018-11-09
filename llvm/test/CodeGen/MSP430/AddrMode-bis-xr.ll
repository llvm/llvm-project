; RUN: llc < %s -march=msp430 | FileCheck %s
target datalayout = "e-p:16:16:16-i8:8:8-i16:16:16-i32:16:16"
target triple = "msp430-generic-generic"

define void @am1(i16* %a, i16 %x) nounwind {
	%1 = load i16, i16* %a
	%2 = or i16 %x, %1
	store i16 %2, i16* %a
	ret void
}
; CHECK-LABEL: am1:
; CHECK:		bis.w	r13, 0(r12)

@foo = external global i16

define void @am2(i16 %x) nounwind {
	%1 = load i16, i16* @foo
	%2 = or i16 %x, %1
	store i16 %2, i16* @foo
	ret void
}
; CHECK-LABEL: am2:
; CHECK:		bis.w	r12, &foo

@bar = external global [2 x i8]

define void @am3(i16 %i, i8 %x) nounwind {
	%1 = getelementptr [2 x i8], [2 x i8]* @bar, i16 0, i16 %i
	%2 = load i8, i8* %1
	%3 = or i8 %x, %2
	store i8 %3, i8* %1
	ret void
}
; CHECK-LABEL: am3:
; CHECK:		bis.b	r13, bar(r12)

define void @am4(i16 %x) nounwind {
	%1 = load volatile i16, i16* inttoptr(i16 32 to i16*)
	%2 = or i16 %x, %1
	store volatile i16 %2, i16* inttoptr(i16 32 to i16*)
	ret void
}
; CHECK-LABEL: am4:
; CHECK:		bis.w	r12, &32

define void @am5(i16* %a, i16 %x) readonly {
	%1 = getelementptr inbounds i16, i16* %a, i16 2
	%2 = load i16, i16* %1
	%3 = or i16 %x, %2
	store i16 %3, i16* %1
	ret void
}
; CHECK-LABEL: am5:
; CHECK:		bis.w	r13, 4(r12)

%S = type { i16, i16 }
@baz = common global %S zeroinitializer

define void @am6(i16 %x) nounwind {
	%1 = load i16, i16* getelementptr (%S, %S* @baz, i32 0, i32 1)
	%2 = or i16 %x, %1
	store i16 %2, i16* getelementptr (%S, %S* @baz, i32 0, i32 1)
	ret void
}
; CHECK-LABEL: am6:
; CHECK:		bis.w	r12, &baz+2

%T = type { i16, [2 x i8] }
@duh = external global %T

define void @am7(i16 %n, i8 %x) nounwind {
	%1 = getelementptr %T, %T* @duh, i32 0, i32 1
	%2 = getelementptr [2 x i8], [2 x i8]* %1, i16 0, i16 %n
	%3 = load i8, i8* %2
	%4 = or i8 %x, %3
	store i8 %4, i8* %2
	ret void
}
; CHECK-LABEL: am7:
; CHECK:		bis.b	r13, duh+2(r12)

