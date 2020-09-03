; RUN: llc -mtriple=arm-eabi -mcpu=cortex-a8 %s -o - | FileCheck %s
; RUN: llc -mtriple=arm-eabi -mattr=+v4t %s -o - | FileCheck %s -check-prefix CHECK-V4-CMP
; RUN: llc -mtriple=arm-eabi -mattr=+v4t %s -o - | FileCheck %s -check-prefix CHECK-V4-BX

define i32 @t1(i32 %a, i32 %b, i32 %c, i32 %d) {
; CHECK-LABEL: t1:
; CHECK: cmp r2, #7
; CHECK: cmpne r2, #1
	switch i32 %c, label %cond_next [
		 i32 1, label %cond_true
		 i32 7, label %cond_true
	]

cond_true:
; CHECK: addne r0
; CHECK: addeq r0
; CHECK: addeq r0
; CHECK: bx
	%tmp12 = add i32 %a, 1
	%tmp1518 = add i32 %tmp12, %b
	ret i32 %tmp1518

cond_next:
	%tmp15 = add i32 %b, %a
	ret i32 %tmp15
}

; CHECK-V4-CMP: cmpne
; CHECK-V4-CMP-NOT: cmpne

; CHECK-V4-BX: bx
; CHECK-V4-BX-NOT: bx

