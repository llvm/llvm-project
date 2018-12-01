; RUN: llc -verify-machineinstrs -enable-machine-outliner -mtriple=aarch64-apple-darwin < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -enable-machine-outliner -mtriple=aarch64-apple-darwin -mcpu=cortex-a53 -enable-misched=false < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -enable-machine-outliner -enable-linkonceodr-outlining -mtriple=aarch64-apple-darwin < %s | FileCheck %s -check-prefix=ODR
; RUN: llc -verify-machineinstrs -enable-machine-outliner -mtriple=aarch64-apple-darwin -stop-after=machine-outliner < %s | FileCheck %s -check-prefix=TARGET_FEATURES

; Make sure that we inherit target features from functions and make sure we have
; the right function attributes.
; TARGET_FEATURES: define internal void @OUTLINED_FUNCTION_{{[0-9]+}}()
; TARGET_FEATURES-SAME: #[[ATTR_NUM:[0-9]+]]
; TARGET_FEATURES-DAG: attributes #[[ATTR_NUM]] = {
; TARGET_FEATURES-SAME: minsize
; TARGET_FEATURES-SAME: optsize
; TARGET_FEATURES-SAME: "target-features"="+sse"

define linkonce_odr void @fish() #0 {
  ; CHECK-LABEL: _fish:
  ; CHECK-NOT: OUTLINED
  ; ODR: [[OUTLINED:OUTLINED_FUNCTION_[0-9]+]]
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
  ret void
}

define void @turtle() section "TURTLE,turtle" {
  ; CHECK-LABEL: _turtle:
  ; ODR-LABEL: _turtle:
  ; CHECK-NOT: OUTLINED
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
  ret void
}

define void @cat() #0 {
  ; CHECK-LABEL: _cat:
  ; CHECK: [[OUTLINED:OUTLINED_FUNCTION_[0-9]+]]
  ; ODR: [[OUTLINED]]
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
  ret void
}

define void @dog() #0 {
  ; CHECK-LABEL: _dog:
  ; CHECK: [[OUTLINED]]
  ; ODR: [[OUTLINED]]
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
  ret void
}

; ODR: [[OUTLINED]]:
; CHECK: .p2align 2
; CHECK-NEXT: [[OUTLINED]]:
; CHECK: orr     w8, wzr, #0x1
; CHECK-NEXT: str     w8, [sp, #28]
; CHECK-NEXT: orr     w8, wzr, #0x2
; CHECK-NEXT: str     w8, [sp, #24]
; CHECK-NEXT: orr     w8, wzr, #0x3
; CHECK-NEXT: str     w8, [sp, #20]
; CHECK-NEXT: orr     w8, wzr, #0x4
; CHECK-NEXT: str     w8, [sp, #16]
; CHECK-NEXT: mov     w8, #5
; CHECK-NEXT: str     w8, [sp, #12]
; CHECK-NEXT: orr     w8, wzr, #0x6
; CHECK-NEXT: str     w8, [sp, #8]
; CHECK-NEXT: add     sp, sp, #32
; CHECK-NEXT: ret

attributes #0 = { noredzone "target-cpu"="cyclone" "target-features"="+sse" }
