; RUN: llc -mtriple=thumb-eabi < %s -o - | FileCheck %s

define void @f1() {
	%c = alloca i8, align 1
	ret void
}

define i32 @f2() {
	ret i32 1
}

; CHECK: add
; CHECK-NOT: add

