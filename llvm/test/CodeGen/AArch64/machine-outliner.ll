; RUN: llc -verify-machineinstrs -enable-machine-outliner -aarch64-load-store-renaming=true -mtriple=aarch64-apple-darwin < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -enable-machine-outliner -aarch64-load-store-renaming=true -mtriple=aarch64-apple-darwin -mcpu=cortex-a53 -enable-misched=false < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -enable-machine-outliner -enable-linkonceodr-outlining -mtriple=aarch64-apple-darwin < %s | FileCheck %s -check-prefix=ODR
; RUN: llc -verify-machineinstrs -enable-machine-outliner -mtriple=aarch64-apple-darwin -stop-after=machine-outliner < %s | FileCheck %s -check-prefix=TARGET_FEATURES

; Make sure that we inherit target features from functions and make sure we have
; the right function attributes.
; TARGET_FEATURES: define internal void @OUTLINED_FUNCTION_{{[0-9]+}}()
; TARGET_FEATURES-SAME: #[[ATTR_NUM:[0-9]+]]
; TARGET_FEATURES-DAG: attributes #[[ATTR_NUM]] = {
; TARGET_FEATURES-SAME: minsize
; TARGET_FEATURES-SAME: nounwind
; TARGET_FEATURES-SAME: optsize
; TARGET_FEATURES-SAME: "target-cpu"="cyclone"
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
  store i32 1, ptr %1, align 4
  store i32 2, ptr %2, align 4
  store i32 3, ptr %3, align 4
  store i32 4, ptr %4, align 4
  store i32 5, ptr %5, align 4
  store i32 6, ptr %6, align 4
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
  store i32 1, ptr %1, align 4
  store i32 2, ptr %2, align 4
  store i32 3, ptr %3, align 4
  store i32 4, ptr %4, align 4
  store i32 5, ptr %5, align 4
  store i32 6, ptr %6, align 4
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
  store i32 1, ptr %1, align 4
  store i32 2, ptr %2, align 4
  store i32 3, ptr %3, align 4
  store i32 4, ptr %4, align 4
  store i32 5, ptr %5, align 4
  store i32 6, ptr %6, align 4
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
  store i32 1, ptr %1, align 4
  store i32 2, ptr %2, align 4
  store i32 3, ptr %3, align 4
  store i32 4, ptr %4, align 4
  store i32 5, ptr %5, align 4
  store i32 6, ptr %6, align 4
  ret void
}

; ODR: [[OUTLINED]]:
; CHECK: .p2align 2
; CHECK-NEXT: [[OUTLINED]]:
; CHECK:      mov     w9, #1
; CHECK-DAG: mov     w8, #2
; CHECK-DAG: stp     w8, w9, [sp, #24]
; CHECK-DAG: mov     w9, #3
; CHECK-DAG: mov     w8, #4
; CHECK-DAG: stp     w8, w9, [sp, #16]
; CHECK-DAG: mov     w9, #5
; CHECK-DAG: mov     w8, #6
; CHECK-DAG: stp     w8, w9, [sp, #8]
; CHECK-DAG: add     sp, sp, #32
; CHECK-DAG: ret

attributes #0 = { nounwind noredzone "target-cpu"="cyclone" "target-features"="+sse" }
