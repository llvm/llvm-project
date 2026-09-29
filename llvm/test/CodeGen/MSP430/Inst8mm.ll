; RUN: llc -mtriple=msp430 < %s | FileCheck %s
target datalayout = "e-p:16:8:8-i8:8:8-i16:8:8-i32:8:8"
target triple = "msp430-generic-generic"

@foo = common global i8 0, align 1
@bar = common global i8 0, align 1

define void @mov() nounwind {
; CHECK-LABEL: mov:
; CHECK: mov.b	&bar, &foo
        %1 = load i8, ptr @bar
        store i8 %1, ptr @foo
        ret void
}

define void @add() nounwind {
; CHECK-LABEL: add:
; CHECK: add.b	&bar, &foo
	%1 = load i8, ptr @bar
	%2 = load i8, ptr @foo
	%3 = add i8 %2, %1
	store i8 %3, ptr @foo
	ret void
}

define void @and() nounwind {
; CHECK-LABEL: and:
; CHECK: and.b	&bar, &foo
	%1 = load i8, ptr @bar
	%2 = load i8, ptr @foo
	%3 = and i8 %2, %1
	store i8 %3, ptr @foo
	ret void
}

define void @bis() nounwind {
; CHECK-LABEL: bis:
; CHECK: bis.b	&bar, &foo
	%1 = load i8, ptr @bar
	%2 = load i8, ptr @foo
	%3 = or i8 %2, %1
	store i8 %3, ptr @foo
	ret void
}

define void @xor() nounwind {
; CHECK-LABEL: xor:
; CHECK: xor.b	&bar, &foo
	%1 = load i8, ptr @bar
	%2 = load i8, ptr @foo
	%3 = xor i8 %2, %1
	store i8 %3, ptr @foo
	ret void
}

define void @cmp(ptr %g, ptr %i) {
entry:
; CHECK-LABEL: cmp:
; CHECK: cmp.b 4(r12), 2(r13)
  %add.ptr = getelementptr inbounds i8, ptr %g, i16 4
  %0 = load i8, ptr %add.ptr, align 1
  %add.ptr1 = getelementptr inbounds i8, ptr %i, i16 2
  %1 = load i8, ptr %add.ptr1, align 1
  %cmp = icmp sgt i8 %0, %1
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i8 0, ptr %g, align 2
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}
