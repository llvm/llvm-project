; RUN: llc -verify-machineinstrs -enable-machine-outliner -mtriple aarch64 %s -o - | \
; RUN:   FileCheck %s --check-prefixes CHECK,V8A
; RUN: llc -verify-machineinstrs -enable-machine-outliner -mtriple aarch64 -mattr=+v8.3a %s -o - | \
; RUN:   FileCheck %s --check-prefixes CHECK,V83A

;; Function a's outlining candidate contains a sp modifying add without a
;; corresponding sub, so we shouldn't outline it.
define void @a() "sign-return-address"="all" "sign-return-address-key"="b_key" {
; CHECK-LABEL:      a:                                      // @a
; CHECK-NOT:        bl OUTLINED_FUNCTION_
; CHECK:            // %bb.0:
; CHECK-NEXT:       .cfi_b_key_frame
; V8A-NEXT:         hint #27
; V83A-NEXT:        pacibsp
; CHECK-NEXT:       .cfi_negate_ra_state
; CHECK-NEXT:       sub sp, sp, #32
; CHECK-NEXT:       .cfi_def_cfa_offset 32
; CHECK-NEXT:       mov w8, #1
; CHECK-NEXT:       mov w9, #2
; CHECK-NEXT:       stp w9, w8, [sp, #24]
; CHECK-NEXT:       mov w9, #3
; CHECK-NEXT:       mov w8, #4
; CHECK-NEXT:       stp w8, w9, [sp, #16]
; CHECK-NEXT:       mov w9, #5
; CHECK-NEXT:       mov w8, #6
; CHECK-NEXT:       stp w8, w9, [sp, #8]
; CHECK-NEXT:       add sp, sp, #32
; V8A-NEXT:         hint #31
; V8A-NEXT:         ret
; V83A-NEXT:        retab
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
  ret void
}

define void @b() "sign-return-address"="all" "sign-return-address-key"="b_key" nounwind {
; CHECK-LABEL:      b:                                      // @b
; CHECK-NEXT:       // %bb.0:
; V8A-NEXT:         hint #27
; V83A-NEXT:        pacibsp
; CHECK-NEXT:       mov x0, x30
; CHECK-NEXT:       bl [[OUTLINED_FUNC:OUTLINED_FUNCTION_[0-9]+]]
; CHECK-NEXT:       mov x30, x0
; V8A-NEXT:         hint #31
; V8A-NEXT:         ret
; V83A-NEXT:        retab
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
  ret void
}

define void @c() "sign-return-address"="all" "sign-return-address-key"="b_key" nounwind {
; CHECK-LABEL:      c:                                      // @c
; CHECK-NEXT:       // %bb.0:
; V8A-NEXT:         hint #27
; V83A-NEXT:        pacibsp
; CHECK-NEXT:       mov x0, x30
; CHECK-NEXT:       bl [[OUTLINED_FUNC]]
; CHECK-NEXT:       mov x30, x0
; V8A-NEXT:         hint #31
; V8A-NEXT:         ret
; V83A-NEXT:        retab
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
  ret void
}

; CHECK:            [[OUTLINED_FUNC]]:
; CHECK:            // %bb.0:
; V8A-NEXT:         hint #27
; V83A-NEXT:        pacibsp
; CHECK-NEXT:       sub sp, sp, #32
; CHECK-NEXT:       mov w8, #1
; CHECK-NEXT:       mov w9, #2
; CHECK-NEXT:       stp w9, w8, [sp, #24]
; CHECK-NEXT:       mov w9, #3
; CHECK-NEXT:       mov w8, #4
; CHECK-NEXT:       stp w8, w9, [sp, #16]
; CHECK-NEXT:       mov w9, #5
; CHECK-NEXT:       mov w8, #6
; CHECK-NEXT:       stp w8, w9, [sp, #8]
; CHECK-NEXT:       add sp, sp, #32
; V8A-NEXT:         hint #31
; V8A-NEXT:         ret
; V83A-NEXT:        retab
