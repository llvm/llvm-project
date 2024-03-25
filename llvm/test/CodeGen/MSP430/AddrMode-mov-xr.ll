; RUN: llc < %s -march=msp430 | FileCheck %s
target datalayout = "e-p:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:16:16"
target triple = "msp430-generic-generic"

define void @am1(ptr %a, i16 %b) nounwind {
	store i16 %b, ptr %a
	ret void
}
; CHECK-LABEL: am1:
; CHECK:		mov	r13, 0(r12)

@foo = external global i16

define void @am2(i16 %a) nounwind {
	store i16 %a, ptr @foo
	ret void
}
; CHECK-LABEL: am2:
; CHECK:		mov	r12, &foo

@bar = external global [2 x i8]

define void @am3(i16 %i, i8 %a) nounwind {
	%1 = getelementptr [2 x i8], ptr @bar, i16 0, i16 %i
	store i8 %a, ptr %1
	ret void
}
; CHECK-LABEL: am3:
; CHECK:		mov.b	r13, bar(r12)

define void @am4(i16 %a) nounwind {
	store volatile i16 %a, ptr inttoptr(i16 32 to ptr)
	ret void
}
; CHECK-LABEL: am4:
; CHECK:		mov	r12, &32

define void @am5(ptr nocapture %p, i16 %a) nounwind readonly {
	%1 = getelementptr inbounds i16, ptr %p, i16 2
	store i16 %a, ptr %1
	ret void
}
; CHECK-LABEL: am5:
; CHECK:		mov	r13, 4(r12)

%S = type { i16, i16 }
@baz = common global %S zeroinitializer, align 1

define void @am6(i16 %a) nounwind {
	store i16 %a, ptr getelementptr (%S, ptr @baz, i32 0, i32 1)
	ret void
}
; CHECK-LABEL: am6:
; CHECK:		mov	r12, &baz+2

%T = type { i16, [2 x i8] }
@duh = external global %T

define void @am7(i16 %n, i8 %a) nounwind {
	%1 = getelementptr %T, ptr @duh, i32 0, i32 1
	%2 = getelementptr [2 x i8], ptr %1, i16 0, i16 %n
	store i8 %a, ptr %2
	ret void
}
; CHECK-LABEL: am7:
; CHECK:		mov.b	r13, duh+2(r12)

