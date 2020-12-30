; RUN: llc %s -mtriple=aarch64-none-linux-gnu -aarch64-enable-global-merge -global-merge-on-external -o - | FileCheck %s
; RUN: llc %s -mtriple=aarch64-linux-gnuabi -aarch64-enable-global-merge -global-merge-on-external -o - | FileCheck %s
; RUN: llc %s -mtriple=aarch64-apple-ios -aarch64-enable-global-merge -global-merge-on-external -o - | FileCheck %s --check-prefix=CHECK-APPLE-IOS

@x = dso_local global i32 0, align 4
@y = dso_local global i32 0, align 4
@z = dso_local global i32 0, align 4

define dso_local void @f1(i32 %a1, i32 %a2) {
;CHECK-APPLE-IOS-LABEL: _f1:
;CHECK-APPLE-IOS-NOT: adrp
;CHECK-APPLE-IOS: adrp	x8, __MergedGlobals_x@PAGE
;CHECK-APPLE-IOS: add	x8, x8, __MergedGlobals_x@PAGEOFF
;CHECK-APPLE-IOS-NOT: adrp
  store i32 %a1, i32* @x, align 4
  store i32 %a2, i32* @y, align 4
  ret void
}

define dso_local void @g1(i32 %a1, i32 %a2) {
;CHECK-APPLE-IOS-LABEL: _g1:
;CHECK-APPLE-IOS: adrp	x8, __MergedGlobals_x@PAGE
;CHECK-APPLE-IOS: add	x8, x8, __MergedGlobals_x@PAGEOFF
;CHECK-APPLE-IOS-NOT: adrp
  store i32 %a1, i32* @y, align 4
  store i32 %a2, i32* @z, align 4
  ret void
}

;CHECK:	.type	.L_MergedGlobals,@object // @_MergedGlobals
;CHECK:	.local	.L_MergedGlobals
;CHECK:	.comm	.L_MergedGlobals,12,4

;CHECK:	.globl	x
;CHECK: .set x, .L_MergedGlobals
;CHECK: .size x, 4
;CHECK:	.globl	y
;CHECK: .set y, .L_MergedGlobals+4
;CHECK: .size y, 4
;CHECK:	.globl	z
;CHECK: .set z, .L_MergedGlobals+8
;CHECK: .size z, 4

;CHECK-APPLE-IOS: .zerofill __DATA,__common,__MergedGlobals_x,12,2

;CHECK-APPLE-IOS: .globl	_x
;CHECK-APPLE-IOS: .set {{.*}}, __MergedGlobals_x
;CHECK-APPLE-IOS: .globl	_y
;CHECK-APPLE-IOS: .set _y, __MergedGlobals_x+4
;CHECK-APPLE-IOS: .globl	_z
;CHECK-APPLE-IOS: .set _z, __MergedGlobals_x+8
;CHECK-APPLE-IOS: .subsections_via_symbols
