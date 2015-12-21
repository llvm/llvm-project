; RUN: llc %s -mtriple=aarch64-none-linux-gnu -aarch64-global-merge -global-merge-on-external -disable-post-ra -o - | FileCheck %s
; RUN: llc %s -mtriple=aarch64-linux-gnuabi -aarch64-global-merge -global-merge-on-external -disable-post-ra -o - | FileCheck %s
; RUN: llc %s -mtriple=aarch64-apple-ios -aarch64-global-merge -global-merge-on-external -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-APPLE-IOS

@x = global [1000 x i32] zeroinitializer, align 1
@y = global [1000 x i32] zeroinitializer, align 1
@z = internal global i32 1, align 4

define void @f1(i32 %a1, i32 %a2, i32 %a3) {
;CHECK-APPLE-IOS: adrp	x8, l__MergedGlobals@PAGE
;CHECK-APPLE-IOS-NOT: adrp
;CHECK-APPLE-IOS: add	x8, x8, l__MergedGlobals@PAGEOFF
;CHECK-APPLE-IOS: adrp	x9, l__MergedGlobals.1@PAGE
;CHECK-APPLE-IOS: add	x9, x9, l__MergedGlobals.1@PAGEOFF
  %x3 = getelementptr inbounds [1000 x i32], [1000 x i32]* @x, i32 0, i64 3
  %y3 = getelementptr inbounds [1000 x i32], [1000 x i32]* @y, i32 0, i64 3
  store i32 %a1, i32* %x3, align 4
  store i32 %a2, i32* %y3, align 4
  store i32 %a3, i32* @z, align 4
  ret void
}

;CHECK:	.type	.L_MergedGlobals,@object // @_MergedGlobals
;CHECK: .align	4
;CHECK: .L_MergedGlobals:
;CHECK: .size	.L_MergedGlobals, 4004

;CHECK: .type	.L_MergedGlobals.1,@object // @_MergedGlobals.1
;CHECK: .local	.L_MergedGlobals.1
;CHECK: .comm	.L_MergedGlobals.1,4000,16

;CHECK-APPLE-IOS: .align	4
;CHECK-APPLE-IOS:  l__MergedGlobals:
;CHECK-APPLE-IOS: .long 1
;CHECK-APPLE-IOS: .space	4000

;CHECK-APPLE-IOS: .zerofill __DATA,__bss,l__MergedGlobals.1,4000,4

;CHECK: z = .L_MergedGlobals
;CHECK:	.globl	x
;CHECK: x = .L_MergedGlobals+4
;CHECK: .size x, 4000
;CHECK:	.globl	y
;CHECK: y = .L_MergedGlobals.1
;CHECK: .size y, 4000

;CHECK-APPLE-IOS-NOT: _z = l__MergedGlobals
;CHECK-APPLE-IOS:.globl	_x
;CHECK-APPLE-IOS: _x = l__MergedGlobals+4
;CHECK-APPLE-IOS:.globl	_y
;CHECK-APPLE-IOS: _y = l__MergedGlobals.1
