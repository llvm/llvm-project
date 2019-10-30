; RUN: llc -verify-machineinstrs -enable-machine-outliner -mtriple \
; RUN: aarch64-arm-none-eabi %s -o - | FileCheck %s

define void @a() "sign-return-address"="all" {
; CHECK-LABEL:      a:                                     // @a
; CHECK:            paciasp
; CHECK-NEXT:       .cfi_negate_ra_state
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  store i32 1, i32* %1, align 4
  store i32 2, i32* %2, align 4
  store i32 3, i32* %3, align 4
  store i32 4, i32* %4, align 4
  store i32 5, i32* %5, align 4
  store i32 6, i32* %6, align 4
; CHECK:          autiasp
  ret void
; CHECK:          .cfi_endproc
}

define void @b() "sign-return-address"="all" "sign-return-address-key"="b_key" {
; CHECK-LABEL:      b:                                     // @b
; CHECK:            .cfi_b_key_frame
; CHECK-NEXT:       pacibsp
; CHECK-NEXT:       .cfi_negate_ra_state
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  store i32 1, i32* %1, align 4
  store i32 2, i32* %2, align 4
  store i32 3, i32* %3, align 4
  store i32 4, i32* %4, align 4
  store i32 5, i32* %5, align 4
  store i32 6, i32* %6, align 4
; CHECK-NOT:        autiasp
  ret void
; CHECK:            .cfi_endproc
}

define void @c() "sign-return-address"="all" {
; CHECK-LABEL:      c:                                     // @c
; CHECK:            paciasp
; CHECK-NEXT:       .cfi_negate_ra_state
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  store i32 1, i32* %1, align 4
  store i32 2, i32* %2, align 4
  store i32 3, i32* %3, align 4
  store i32 4, i32* %4, align 4
  store i32 5, i32* %5, align 4
  store i32 6, i32* %6, align 4
; CHECK:          autiasp
  ret void
; CHECK:          .cfi_endproc
}

; CHECK-NOT:      OUTLINED_FUNCTION_0:
; CHECK-NOT:      // -- Begin function
