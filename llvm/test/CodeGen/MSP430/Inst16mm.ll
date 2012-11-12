; RUN: llc -march=msp430 -combiner-alias-analysis < %s | FileCheck %s
target datalayout = "e-p:16:8:8-i8:8:8-i16:8:8-i32:8:8"
target triple = "msp430-generic-generic"
@foo = common global i16 0, align 2
@bar = common global i16 0, align 2

define void @mov() nounwind {
; CHECK: mov:
; CHECK: mov.w	&bar, &foo
        %1 = load i16* @bar
        store i16 %1, i16* @foo
        ret void
}

define void @add() nounwind {
; CHECK: add:
; CHECK: add.w	&bar, &foo
	%1 = load i16* @bar
	%2 = load i16* @foo
	%3 = add i16 %2, %1
	store i16 %3, i16* @foo
	ret void
}

define void @and() nounwind {
; CHECK: and:
; CHECK: and.w	&bar, &foo
	%1 = load i16* @bar
	%2 = load i16* @foo
	%3 = and i16 %2, %1
	store i16 %3, i16* @foo
	ret void
}

define void @bis() nounwind {
; CHECK: bis:
; CHECK: bis.w	&bar, &foo
	%1 = load i16* @bar
	%2 = load i16* @foo
	%3 = or i16 %2, %1
	store i16 %3, i16* @foo
	ret void
}

define void @xor() nounwind {
; CHECK: xor:
; CHECK: xor.w	&bar, &foo
	%1 = load i16* @bar
	%2 = load i16* @foo
	%3 = xor i16 %2, %1
	store i16 %3, i16* @foo
	ret void
}

define i16 @mov2() nounwind {
entry:
 %retval = alloca i16                            ; <i16*> [#uses=3]
 %x = alloca i32, align 2                        ; <i32*> [#uses=1]
 %y = alloca i32, align 2                        ; <i32*> [#uses=1]
 store i16 0, i16* %retval
 %tmp = load i32* %y                             ; <i32> [#uses=1]
 store i32 %tmp, i32* %x
 store i16 0, i16* %retval
 %0 = load i16* %retval                          ; <i16> [#uses=1]
 ret i16 %0
; CHECK: mov2:
; CHECK:	mov.w	2(r1), 6(r1)
; CHECK:	mov.w	0(r1), 4(r1)
}
