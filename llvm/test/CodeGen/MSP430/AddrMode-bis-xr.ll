; RUN: llc < %s -mtriple=msp430 | FileCheck %s
target datalayout = "e-p:16:16:16-i8:8:8-i16:16:16-i32:16:16"
target triple = "msp430-generic-generic"

define void @am1(ptr %a, i16 %x) nounwind {
	%1 = load i16, ptr %a
	%2 = or i16 %x, %1
	store i16 %2, ptr %a
	ret void
}
; CHECK-LABEL: am1:
; CHECK:		bis	r13, 0(r12)

@foo = external global i16

define void @am2(i16 %x) nounwind {
	%1 = load i16, ptr @foo
	%2 = or i16 %x, %1
	store i16 %2, ptr @foo
	ret void
}
; CHECK-LABEL: am2:
; CHECK:		bis	r12, &foo

@bar = external global [2 x i8]

define void @am3(i16 %i, i8 %x) nounwind {
	%1 = getelementptr [2 x i8], ptr @bar, i16 0, i16 %i
	%2 = load i8, ptr %1
	%3 = or i8 %x, %2
	store i8 %3, ptr %1
	ret void
}
; CHECK-LABEL: am3:
; CHECK:		bis.b	r13, bar(r12)

define void @am4(i16 %x) nounwind {
	%1 = load volatile i16, ptr inttoptr(i16 32 to ptr)
	%2 = or i16 %x, %1
	store volatile i16 %2, ptr inttoptr(i16 32 to ptr)
	ret void
}
; CHECK-LABEL: am4:
; CHECK:		bis	r12, &32

define void @am5(ptr %a, i16 %x) readonly {
	%1 = getelementptr inbounds i16, ptr %a, i16 2
	%2 = load i16, ptr %1
	%3 = or i16 %x, %2
	store i16 %3, ptr %1
	ret void
}
; CHECK-LABEL: am5:
; CHECK:		bis	r13, 4(r12)

%S = type { i16, i16 }
@baz = common global %S zeroinitializer

define void @am6(i16 %x) nounwind {
	%1 = load i16, ptr getelementptr (%S, ptr @baz, i32 0, i32 1)
	%2 = or i16 %x, %1
	store i16 %2, ptr getelementptr (%S, ptr @baz, i32 0, i32 1)
	ret void
}
; CHECK-LABEL: am6:
; CHECK:		bis	r12, &baz+2

%T = type { i16, [2 x i8] }
@duh = external global %T

define void @am7(i16 %n, i8 %x) nounwind {
	%1 = getelementptr %T, ptr @duh, i32 0, i32 1
	%2 = getelementptr [2 x i8], ptr %1, i16 0, i16 %n
	%3 = load i8, ptr %2
	%4 = or i8 %x, %3
	store i8 %4, ptr %2
	ret void
}
; CHECK-LABEL: am7:
; CHECK:		bis.b	r13, duh+2(r12)

