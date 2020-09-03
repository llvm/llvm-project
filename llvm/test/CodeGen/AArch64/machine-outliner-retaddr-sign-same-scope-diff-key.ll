; RUN: llc -verify-machineinstrs -enable-machine-outliner -mtriple \
; RUN: aarch64-arm-none-eabi %s -o - | FileCheck %s --check-prefixes CHECK,V8A
; RUN-V83A: llc -verify-machineinstrs -enable-machine-outliner -mtriple \
; RUN-V83A: aarch64-arm-none-eabi -mattr=+v8.3a %s -o - > %t
; RUN-V83A: FileCheck --check-prefixes CHECK,V83A < %t %s

define void @a() "sign-return-address"="all" {
; CHECK-LABEL:      a:                                     // @a
; V8A:              hint #25
; V83A:             paciasp
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
; V8A:            hint #29
; V83A:           autiasp
  ret void
; CHECK:          .cfi_endproc
}

define void @b() "sign-return-address"="all" "sign-return-address-key"="b_key" {
; CHECK-LABEL:      b:                                     // @b
; CHECK:            .cfi_b_key_frame
; V8A-NEXT:         hint #27
; V83A-NEXT:        pacibsp
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
; V8A-NOT:          hint #29
; V83A-NOT:         autiasp
  ret void
; CHECK:            .cfi_endproc
}

define void @c() "sign-return-address"="all" {
; CHECK-LABEL:      c:                                     // @c
; V8A:              hint #25
; V83A:             paciasp
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
; V8A:            hint #29
; V83A:           autiasp
  ret void
; CHECK:          .cfi_endproc
}

; CHECK-NOT:      OUTLINED_FUNCTION_0:
; CHECK-NOT:      // -- Begin function
