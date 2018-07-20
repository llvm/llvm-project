; RUN: llc < %s -mtriple=arm-linux-gnueabi -asm-verbose=false -verify-machineinstrs | FileCheck %s -check-prefix=CHECK-ARM
; RUN: llc < %s -mtriple=thumb-linux-gnueabi -asm-verbose=false -verify-machineinstrs | FileCheck %s -check-prefix=CHECK-THUMB

; RUN: llc < %s -mtriple=armv6-linux-gnueabi -asm-verbose=false -verify-machineinstrs | FileCheck %s -check-prefix=CHECK-ARMV6
; RUN: llc < %s -mtriple=thumbv6-linux-gnueabi -asm-verbose=false -verify-machineinstrs | FileCheck %s -check-prefix=CHECK-THUMBV6

; RUN: llc < %s -mtriple=armv7-linux-gnueabi -asm-verbose=false -verify-machineinstrs | FileCheck %s -check-prefix=CHECK-ARMV7
; RUN: llc < %s -mtriple=thumbv7-linux-gnueabi -asm-verbose=false -verify-machineinstrs | FileCheck %s -check-prefix=CHECK-THUMBV7

define zeroext i1 @test_cmpxchg_res_i8(i8* %addr, i8 %desired, i8 zeroext %new) {
entry:
  %0 = cmpxchg i8* %addr, i8 %desired, i8 %new monotonic monotonic
  %1 = extractvalue { i8, i1 } %0, 1
  ret i1 %1
}

; CHECK-ARM-LABEL: test_cmpxchg_res_i8
; CHECK-ARM: bl __sync_val_compare_and_swap_1
; CHECK-ARM: sub r0, r0, {{r[0-9]+}}
; CHECK-ARM: rsbs [[REG:r[0-9]+]], r0, #0
; CHECK-ARM: adc r0, r0, [[REG]]

; CHECK-THUMB-LABEL: test_cmpxchg_res_i8
; CHECK-THUMB: bl __sync_val_compare_and_swap_1
; CHECK-THUMB-NOT: mov [[R1:r[0-7]]], r0
; CHECK-THUMB: subs [[R1:r[0-7]]], r0, {{r[0-9]+}}
; CHECK-THUMB: movs r0, #0
; CHECK-THUMB: subs r0, r0, [[R1]]
; CHECK-THUMB: adcs r0, [[R1]]

; CHECK-ARMV6-LABEL: test_cmpxchg_res_i8:
; CHECK-ARMV6-NEXT:  .fnstart
; CHECK-ARMV6-NEXT: uxtb [[DESIRED:r[0-9]+]], r1
; CHECK-ARMV6-NEXT: [[TRY:.LBB[0-9_]+]]:
; CHECK-ARMV6-NEXT: ldrexb [[LD:r[0-9]+]], [r0]
; CHECK-ARMV6-NEXT: cmp [[LD]], [[DESIRED]]
; CHECK-ARMV6-NEXT: movne [[RES:r[0-9]+]], #0
; CHECK-ARMV6-NEXT: bxne lr
; CHECK-ARMV6-NEXT: strexb [[SUCCESS:r[0-9]+]], r2, [r0]
; CHECK-ARMV6-NEXT: cmp [[SUCCESS]], #0
; CHECK-ARMV6-NEXT: moveq [[RES]], #1
; CHECK-ARMV6-NEXT: bxeq lr
; CHECK-ARMV6-NEXT: b [[TRY]]

; CHECK-THUMBV6-LABEL: test_cmpxchg_res_i8:
; CHECK-THUMBV6:       mov [[EXPECTED:r[0-9]+]], r1
; CHECK-THUMBV6-NEXT:  bl __sync_val_compare_and_swap_1
; CHECK-THUMBV6-NEXT:  uxtb r1, r4
; CHECK-THUMBV6-NEXT:  subs [[R1:r[0-7]]], r0, {{r[0-9]+}}
; CHECK-THUMBV6-NEXT:  movs r0, #0
; CHECK-THUMBV6-NEXT:  subs r0, r0, [[R1]]
; CHECK-THUMBV6-NEXT:  adcs r0, [[R1]]

; CHECK-ARMV7-LABEL: test_cmpxchg_res_i8:
; CHECK-ARMV7-NEXT: .fnstart
; CHECK-ARMV7-NEXT: uxtb [[DESIRED:r[0-9]+]], r1
; CHECK-ARMV7-NEXT: b [[TRY:.LBB[0-9_]+]]
; CHECK-ARMV7-NEXT: [[HEAD:.LBB[0-9_]+]]:
; CHECK-ARMV7-NEXT: strexb [[SUCCESS:r[0-9]+]], r2, [r0]
; CHECK-ARMV7-NEXT: cmp [[SUCCESS]], #0
; CHECK-ARMV7-NEXT: moveq r0, #1
; CHECK-ARMV7-NEXT: bxeq lr
; CHECK-ARMV7-NEXT: [[TRY]]:
; CHECK-ARMV7-NEXT: ldrexb [[SUCCESS]], [r0]
; CHECK-ARMV7-NEXT: cmp [[SUCCESS]], r1
; CHECK-ARMV7-NEXT: beq [[HEAD]]
; CHECK-ARMV7-NEXT: mov r0, #0
; CHECK-ARMV7-NEXT: clrex
; CHECK-ARMV7-NEXT: bx lr

; CHECK-THUMBV7-LABEL: test_cmpxchg_res_i8:
; CHECK-THUMBV7-NEXT: .fnstart
; CHECK-THUMBV7-NEXT: uxtb [[DESIRED:r[0-9]+]], r1
; CHECK-THUMBV7-NEXT: b [[TRYLD:.LBB[0-9_]+]]
; CHECK-THUMBV7-NEXT: [[TRYST:.LBB[0-9_]+]]:
; CHECK-THUMBV7-NEXT: strexb [[SUCCESS:r[0-9]+]], r2, [r0]
; CHECK-THUMBV7-NEXT: cmp [[SUCCESS]], #0
; CHECK-THUMBV7-NEXT: itt eq
; CHECK-THUMBV7-NEXT: moveq r0, #1
; CHECK-THUMBV7-NEXT: bxeq lr
; CHECK-THUMBV7-NEXT: [[TRYLD]]:
; CHECK-THUMBV7-NEXT: ldrexb [[LD:r[0-9]+]], [r0]
; CHECK-THUMBV7-NEXT: cmp [[LD]], [[DESIRED]]
; CHECK-THUMBV7-NEXT: beq [[TRYST:.LBB[0-9_]+]]
; CHECK-THUMBV7-NEXT: movs r0, #0
; CHECK-THUMBV7-NEXT: clrex
; CHECK-THUMBV7-NEXT: bx lr
