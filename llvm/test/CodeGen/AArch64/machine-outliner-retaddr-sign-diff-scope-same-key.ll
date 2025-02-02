; RUN: llc -verify-machineinstrs -enable-machine-outliner -mtriple aarch64 %s -o - | \
; RUN:   FileCheck %s --check-prefixes CHECK,V8A
; RUN: llc -verify-machineinstrs -enable-machine-outliner -mtriple aarch64 -mattr=+v8.3a %s -o - | \
; RUN:   FileCheck %s --check-prefixes CHECK,V83A

define void @a() "sign-return-address"="all" {
; CHECK-LABEL:      a:                                     // @a
; CHECK:      .cfi_negate_ra_state
; V8A-NEXT:              hint #25
; V83A-NEXT:             paciasp
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  store i32 1, ptr %1, align 4
  store i32 2, ptr %2, align 4
  store i32 3, ptr %3, align 4
  store i32 4, ptr %4, align 4
  store i32 5, ptr %5, align 4
  store i32 6, ptr %6, align 4
; V8A:            hint #29
; V83A:           retaa
  ret void
; CHECK:          .cfi_endproc
}

define void @b() "sign-return-address"="non-leaf" {
; CHECK-LABEL:     b:                                     // @b
; V8A-NOT:         hint #25
; V83A-NOT:        paciasp
; CHECK-NOT:       .cfi_negate_ra_state
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  store i32 1, ptr %1, align 4
  store i32 2, ptr %2, align 4
  store i32 3, ptr %3, align 4
  store i32 4, ptr %4, align 4
  store i32 5, ptr %5, align 4
  store i32 6, ptr %6, align 4
; V8A-NOT:          hint #29
; V83A-NOT:         autiasp
; V83A-NOT:         retaa
  ret void
; CHECK:            .cfi_endproc
}

define void @c() "sign-return-address"="all" {
; CHECK-LABEL:         c:              // @c
; CHECK:      .cfi_negate_ra_state
; V8A-NEXT:              hint #25
; V83A-NEXT:             paciasp
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  store i32 1, ptr %1, align 4
  store i32 2, ptr %2, align 4
  store i32 3, ptr %3, align 4
  store i32 4, ptr %4, align 4
  store i32 5, ptr %5, align 4
  store i32 6, ptr %6, align 4
; V8A:            hint #29
; V83A:           retaa
  ret void
; CHECK:          .cfi_endproc
}

; CHECK-NOT:      OUTLINED_FUNCTION_{{[0-9]+}}:
; CHECK-NOT:      // -- Begin function
