; RUN: llc < %s -march=msp430 | FileCheck %s
target datalayout = "e-p:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:16:16"
target triple = "msp430-generic-generic"

define i16 @am1(i16 %x, i16* %a) nounwind {
	%1 = load i16* %a
	%2 = or i16 %1,%x
	ret i16 %2
}
; CHECK: am1:
; CHECK:		bis.w	0(r14), r15

@foo = external global i16

define i16 @am2(i16 %x) nounwind {
	%1 = load i16* @foo
	%2 = or i16 %1,%x
	ret i16 %2
}
; CHECK: am2:
; CHECK:		bis.w	&foo, r15

@bar = internal constant [2 x i8] [ i8 32, i8 64 ]

define i8 @am3(i8 %x, i16 %n) nounwind {
	%1 = getelementptr [2 x i8]* @bar, i16 0, i16 %n
	%2 = load i8* %1
	%3 = or i8 %2,%x
	ret i8 %3
}
; CHECK: am3:
; CHECK:		bis.b	bar(r14), r15

define i16 @am4(i16 %x) nounwind {
	%1 = load volatile i16* inttoptr(i16 32 to i16*)
	%2 = or i16 %1,%x
	ret i16 %2
}
; CHECK: am4:
; CHECK:		bis.w	&32, r15

define i16 @am5(i16 %x, i16* %a) nounwind {
	%1 = getelementptr i16* %a, i16 2
	%2 = load i16* %1
	%3 = or i16 %2,%x
	ret i16 %3
}
; CHECK: am5:
; CHECK:		bis.w	4(r14), r15

%S = type { i16, i16 }
@baz = common global %S zeroinitializer, align 1

define i16 @am6(i16 %x) nounwind {
	%1 = load i16* getelementptr (%S* @baz, i32 0, i32 1)
	%2 = or i16 %1,%x
	ret i16 %2
}
; CHECK: am6:
; CHECK:		bis.w	&baz+2, r15

%T = type { i16, [2 x i8] }
@duh = internal constant %T { i16 16, [2 x i8][i8 32, i8 64 ] }

define i8 @am7(i8 %x, i16 %n) nounwind {
	%1 = getelementptr %T* @duh, i32 0, i32 1
	%2 = getelementptr [2 x i8]* %1, i16 0, i16 %n
	%3= load i8* %2
	%4 = or i8 %3,%x
	ret i8 %4
}
; CHECK: am7:
; CHECK:		bis.b	duh+2(r14), r15

