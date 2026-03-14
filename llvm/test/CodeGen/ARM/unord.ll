; RUN: llc -mtriple=arm-eabi %s -o - | FileCheck %s

define i32 @f1(float %X, float %Y) {
	%tmp = fcmp uno float %X, %Y
	%retval = select i1 %tmp, i32 1, i32 -1
	ret i32 %retval
}

define i32 @f2(float %X, float %Y) {
	%tmp = fcmp ord float %X, %Y
	%retval = select i1 %tmp, i32 1, i32 -1
	ret i32 %retval
}

; CHECK: movne
; CHECK-NOT: movne

; CHECK: moveq
; CHECK-NOT: moveq

