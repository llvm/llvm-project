; RUN: llc -mtriple=arm-eabi -mcpu=arm1156t2-s -mattr=+thumb2 %s -o - | FileCheck %s
; rdar://7317664

; RUN: llc -mtriple=thumbv8m.base %s -o - | FileCheck %s
; RUN: llc -mtriple=thumbv8m.base -mcpu=cortex-m23 %s -o - | FileCheck %s --check-prefix=NOMOVT
; RUN: llc -mtriple=thumbv8m.base -mcpu=cortex-m33 %s -o - | FileCheck %s
; RUN: llc -mtriple=thumbv8m.base -mcpu=cortex-m35p %s -o - | FileCheck %s

define i32 @t(i32 %X) nounwind {
; CHECK-LABEL: t:
; CHECK: movt r{{[0-9]}}, #65535
; NOMOVT-LABEL: t:
; NOMOVT-NOT: movt r{{[0-9]}}, #65535
; NOMOVT: ldr r{{[0-9]}}, .LCP
entry:
	%0 = or i32 %X, -65536
	ret i32 %0
}

define i32 @t2(i32 %X) nounwind {
; CHECK-LABEL: t2:
; CHECK: movt r{{[0-9]}}, #65534
; NOMOVT-LABEL: t2:
; NOMOVT-NOT: movt r{{[0-9]}}, #65534
; NOMOVT: ldr r{{[0-9]}}, .LCP
entry:
	%0 = or i32 %X, -131072
	%1 = and i32 %0, -65537
	ret i32 %1
}
