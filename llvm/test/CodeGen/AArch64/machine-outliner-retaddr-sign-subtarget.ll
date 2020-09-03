; RUN: llc -verify-machineinstrs -enable-machine-outliner -mtriple \
; RUN: aarch64-arm-linux-gnu %s -o - | FileCheck %s

; Check that functions that should sign their return addresses don't get
; outlined if not all of the function either support v8.3a features or all of
; the functions don't!!

define void @a() #0 {
; CHECK-LABEL:      a:                                     // @a
; CHECK:            // %bb.0:
; CHECK-NEXT:               .cfi_b_key_frame
; CHECK-NEXT:               pacibsp
; CHECK-NEXT:               .cfi_negate_ra_state
; CHECK-NOT:                OUTLINED_FUNCTION_
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
; CHECK:                  retab
; CHECK-NOT:              auti[a,b]sp
  ret void
}

define void @b() #0 {
; CHECK-LABEL:      b:                                     // @b
; CHECK:            // %bb.0:
; CHECK-NEXT:               .cfi_b_key_frame
; CHECK-NEXT:               pacibsp
; CHECK-NEXT:               .cfi_negate_ra_state
; CHECK-NOT:                OUTLINED_FUNCTION_
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
; CHECK:                  retab
; CHECK-NOT:              auti[a,b]sp
  ret void
}

define void @c() #1 {
; CHECK-LABEL:      c:                                     // @c
; CHECK:            // %bb.0:
; CHECK-NEXT:               .cfi_b_key_frame
; CHECK-NEXT:               hint #27
; CHECK-NEXT:               .cfi_negate_ra_state
; CHECK-NOT:                OUTLINED_FUNCTION_
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
; CHECK:                  hint #31
; CHECK-NOT:              ret{{[a,b]}}
  ret void
}

attributes #0 = { "sign-return-address"="all"
                  "sign-return-address-key"="b_key"
                  "target-features"="+v8.3a" }

attributes #1 = { "sign-return-address"="all"
                  "sign-return-address-key"="b_key" }

; CHECK-NOT:                OUTLINED_FUNCTION_
