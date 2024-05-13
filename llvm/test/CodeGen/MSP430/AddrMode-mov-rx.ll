; RUN: llc < %s -march=msp430 | FileCheck %s
target datalayout = "e-p:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:16:16"
target triple = "msp430-generic-generic"

define i16 @am1(ptr %a) nounwind {
	%1 = load i16, ptr %a
	ret i16 %1
}
; CHECK-LABEL: am1:
; CHECK:		mov	0(r12), r12

@foo = external global i16

define i16 @am2() nounwind {
	%1 = load i16, ptr @foo
	ret i16 %1
}
; CHECK-LABEL: am2:
; CHECK:		mov	&foo, r12

@bar = internal constant [2 x i8] [ i8 32, i8 64 ]

define i8 @am3(i16 %n) nounwind {
	%1 = getelementptr [2 x i8], ptr @bar, i16 0, i16 %n
	%2 = load i8, ptr %1
	ret i8 %2
}
; CHECK-LABEL: am3:
; CHECK:		mov.b	bar(r12), r12

define i16 @am4() nounwind {
	%1 = load volatile i16, ptr inttoptr(i16 32 to ptr)
	ret i16 %1
}
; CHECK-LABEL: am4:
; CHECK:		mov	&32, r12

define i16 @am5(ptr %a) nounwind {
	%1 = getelementptr i16, ptr %a, i16 2
	%2 = load i16, ptr %1
	ret i16 %2
}
; CHECK-LABEL: am5:
; CHECK:		mov	4(r12), r12

%S = type { i16, i16 }
@baz = common global %S zeroinitializer, align 1

define i16 @am6() nounwind {
	%1 = load i16, ptr getelementptr (%S, ptr @baz, i32 0, i32 1)
	ret i16 %1
}
; CHECK-LABEL: am6:
; CHECK:		mov	&baz+2, r12

%T = type { i16, [2 x i8] }
@duh = internal constant %T { i16 16, [2 x i8][i8 32, i8 64 ] }

define i8 @am7(i16 %n) nounwind {
	%1 = getelementptr %T, ptr @duh, i32 0, i32 1
	%2 = getelementptr [2 x i8], ptr %1, i16 0, i16 %n
	%3= load i8, ptr %2
	ret i8 %3
}
; CHECK-LABEL: am7:
; CHECK:		mov.b	duh+2(r12), r12

