; RUN: llc -march=msp430 < %s | FileCheck %s
target datalayout = "e-p:16:8:8-i8:8:8-i16:8:8-i32:8:8"
target triple = "msp430-generic-generic"
@foo = common global i16 0, align 2
@bar = common global i16 0, align 2

define void @mov() nounwind {
; CHECK-LABEL: mov:
; CHECK: mov	&bar, &foo
        %1 = load i16, i16* @bar
        store i16 %1, i16* @foo
        ret void
}

define void @add() nounwind {
; CHECK-LABEL: add:
; CHECK: add	&bar, &foo
	%1 = load i16, i16* @bar
	%2 = load i16, i16* @foo
	%3 = add i16 %2, %1
	store i16 %3, i16* @foo
	ret void
}

define void @and() nounwind {
; CHECK-LABEL: and:
; CHECK: and	&bar, &foo
	%1 = load i16, i16* @bar
	%2 = load i16, i16* @foo
	%3 = and i16 %2, %1
	store i16 %3, i16* @foo
	ret void
}

define void @bis() nounwind {
; CHECK-LABEL: bis:
; CHECK: bis	&bar, &foo
	%1 = load i16, i16* @bar
	%2 = load i16, i16* @foo
	%3 = or i16 %2, %1
	store i16 %3, i16* @foo
	ret void
}

define void @xor() nounwind {
; CHECK-LABEL: xor:
; CHECK: xor	&bar, &foo
	%1 = load i16, i16* @bar
	%2 = load i16, i16* @foo
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
 %tmp = load i32, i32* %y                             ; <i32> [#uses=1]
 store i32 %tmp, i32* %x
 store i16 0, i16* %retval
 %0 = load i16, i16* %retval                          ; <i16> [#uses=1]
 ret i16 %0
; CHECK-LABEL: mov2:
; CHECK-DAG:	mov	2(r1), 6(r1)
; CHECK-DAG:	mov	0(r1), 4(r1)
}

define void @cmp(i16* %g, i16* %i) {
entry:
; CHECK-LABEL: cmp:
; CHECK: cmp 8(r12), 4(r13)
  %add.ptr = getelementptr inbounds i16, i16* %g, i16 4
  %0 = load i16, i16* %add.ptr, align 2
  %add.ptr1 = getelementptr inbounds i16, i16* %i, i16 2
  %1 = load i16, i16* %add.ptr1, align 2
  %cmp = icmp sgt i16 %0, %1
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i16 0, i16* %g, align 2
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}
