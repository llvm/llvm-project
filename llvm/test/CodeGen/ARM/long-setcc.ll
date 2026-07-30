; RUN: llc -mtriple=arm-eabi < %s | FileCheck %s

define i1 @t1(i64 %x) {
; CHECK-LABEL: t1:
; CHECK: lsr	r0, r1, #31
	%B = icmp slt i64 %x, 0
	ret i1 %B
}

define i1 @t2(i64 %x) {
; CHECK-LABEL: t2:
; CHECK: rsbs	r0, r1, #0
; CHECK: adc	r0, r1, r0
	%tmp = icmp ult i64 %x, 4294967296
	ret i1 %tmp
}

define i1 @t3(i32 %x) {
; CHECK-LABEL: t3:
; CHECK: mov	r0, #0
	%tmp = icmp ugt i32 %x, -1
	ret i1 %tmp
}

; CHECK-NOT: cmp

