; RUN: llc -verify-machineinstrs -enable-machine-outliner -mtriple \
; RUN: aarch64 %s -o - | FileCheck %s --check-prefixes CHECK,V8A
; RUN-V83A: llc -verify-machineinstrs -enable-machine-outliner -mtriple \
; RUN-V83A: aarch64 -mattr=+v8.3a %s -o - > %t
; RUN-V83A: FileCheck --check-prefixes CHECK,V83A < %t %s

define i64 @a(i64 %x) "sign-return-address"="non-leaf" "sign-return-address-key"="b_key" {
; CHECK-LABEL:      a:                                     // @a
; CHECK:                .cfi_b_key_frame
; V8A-NEXT:             hint #27
; V83A-NEXT:            pacibsp
; CHECK-NEXT:           .cfi_negate_ra_state
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
  call void asm sideeffect "mov x30, $0", "r,~{lr}"(i64 %x) #1
  ret i64 %x
}

define i64 @b(i64 %x) "sign-return-address"="non-leaf" "sign-return-address-key"="b_key" {
; CHECK-LABEL:      b:                                     // @b
; CHECK:                .cfi_b_key_frame
; V8A-NEXT:             hint #27
; V83A-NEXT:            pacibsp
; CHECK-NEXT:           .cfi_negate_ra_state
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
  call void asm sideeffect "mov x30, $0", "r,~{lr}"(i64 %x) #1
  ret i64 %x
}

define i64 @c(i64 %x) "sign-return-address"="non-leaf" "sign-return-address-key"="b_key" {
; CHECK-LABEL:      c:                                     // @c
; CHECK:                .cfi_b_key_frame
; V8A-NEXT:             hint #27
; V83A-NEXT:            pacibsp
; CHECK-NEXT:           .cfi_negate_ra_state
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
  call void asm sideeffect "mov x30, $0", "r,~{lr}"(i64 %x) #1
  ret i64 %x
}

; Outlined function is leaf-function => don't sign it
; CHECK-LABEL:      OUTLINED_FUNCTION_0:
; CHECK-NOT:            .cfi_b_key_frame
; CHECK-NOT:            paci{{[a,b]}}sp
; CHECK-NOT:            hint #2{{[5,7]}}
; CHECK-NOT:            .cfi_negate_ra_state
; CHECK-NOT:            auti{{[a,b]}}sp
