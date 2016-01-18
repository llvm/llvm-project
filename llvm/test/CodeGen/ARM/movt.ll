; RUN: llc -mtriple=arm-eabi -mcpu=arm1156t2-s -mattr=+thumb2 %s -o - | FileCheck %s
; rdar://7317664

define i32 @t(i32 %X) nounwind {
; CHECK-LABEL: t:
; CHECK: movt r0, #65535
entry:
	%0 = or i32 %X, -65536
	ret i32 %0
}

define i32 @t2(i32 %X) nounwind {
; CHECK-LABEL: t2:
; CHECK: movt r0, #65534
entry:
	%0 = or i32 %X, -131072
	%1 = and i32 %0, -65537
	ret i32 %1
}
