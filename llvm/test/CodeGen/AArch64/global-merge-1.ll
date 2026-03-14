; RUN: llc %s -mtriple=aarch64-none-linux-gnu -aarch64-enable-global-merge -o - | FileCheck %s
; RUN: llc %s -mtriple=aarch64-none-linux-gnu -aarch64-enable-global-merge -global-merge-on-external -o - | FileCheck %s

; RUN: llc %s -mtriple=aarch64-linux-gnuabi -aarch64-enable-global-merge -o - | FileCheck %s
; RUN: llc %s -mtriple=aarch64-linux-gnuabi -aarch64-enable-global-merge -global-merge-on-external -o - | FileCheck %s

; RUN: llc %s -mtriple=aarch64-apple-ios -aarch64-enable-global-merge -o - | FileCheck %s --check-prefix=CHECK-APPLE-IOS
; RUN: llc %s -mtriple=aarch64-apple-ios -aarch64-enable-global-merge -global-merge-on-external -o - | FileCheck %s --check-prefix=CHECK-APPLE-IOS

@m = internal global i32 0, align 4
@n = internal global i32 0, align 4

define void @f1(i32 %a1, i32 %a2) {
;CHECK-APPLE-IOS-NOT: adrp
;CHECK-APPLE-IOS: adrp	x8, __MergedGlobals@PAGE
;CHECK-APPLE-IOS-NOT: adrp
;CHECK-APPLE-IOS: add	x8, x8, __MergedGlobals@PAGEOFF
  store i32 %a1, ptr @m, align 4
  store i32 %a2, ptr @n, align 4
  ret void
}

;CHECK:	.type	.L_MergedGlobals,@object  // @_MergedGlobals
;CHECK:	.local	.L_MergedGlobals
;CHECK:	.comm	.L_MergedGlobals,8,4
;CHECK: m = .L_MergedGlobals
;CHECK: n = .L_MergedGlobals+4

;CHECK-APPLE-IOS: .zerofill __DATA,__bss,__MergedGlobals,8,2 ; @_MergedGlobals
;CHECK-APPLE-IOS-NOT: _m = l__MergedGlobals
;CHECK-APPLE-IOS-NOT: _n = l__MergedGlobals+4
