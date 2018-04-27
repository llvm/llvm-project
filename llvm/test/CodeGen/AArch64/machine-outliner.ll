; RUN: llc -verify-machineinstrs -enable-machine-outliner -mtriple=aarch64-apple-darwin < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -enable-machine-outliner -enable-linkonceodr-outlining -mtriple=aarch64-apple-darwin < %s | FileCheck %s -check-prefix=ODR

define linkonce_odr void @fish() #0 {
  ; CHECK-LABEL: _fish:
  ; CHECK-NOT: OUTLINED
  ; ODR: [[OUTLINED:OUTLINED_FUNCTION_[0-9]+]]
  ; ODR-NOT: ret
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  store i32 1, i32* %2, align 4
  store i32 2, i32* %3, align 4
  store i32 3, i32* %4, align 4
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
  store i32 0, i32* %1, align 4
  store i32 1, i32* %2, align 4
  store i32 2, i32* %3, align 4
  store i32 3, i32* %4, align 4
  ret void
}

define void @cat() #0 {
  ; CHECK-LABEL: _cat:
  ; CHECK: [[OUTLINED:OUTLINED_FUNCTION_[0-9]+]]
  ; ODR: [[OUTLINED]]
  ; CHECK-NOT: ret
  ; ODR-NOT: ret
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  store i32 1, i32* %2, align 4
  store i32 2, i32* %3, align 4
  store i32 3, i32* %4, align 4
  ret void
}

define void @dog() #0 {
  ; CHECK-LABEL: _dog:
  ; CHECK: [[OUTLINED]]
  ; ODR: [[OUTLINED]]
  ; CHECK-NOT: ret
  ; ODR-NOT: ret
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  store i32 1, i32* %2, align 4
  store i32 2, i32* %3, align 4
  store i32 3, i32* %4, align 4
  ret void
}

; ODR: [[OUTLINED]]:
; CHECK: [[OUTLINED]]:
; CHECK-DAG: orr w8, wzr, #0x1
; CHECK-NEXT: stp w8, wzr, [sp, #8]
; CHECK-NEXT: orr w8, wzr, #0x2
; CHECK-NEXT: str w8, [sp, #4]
; CHECK-NEXT: orr w8, wzr, #0x3
; CHECK-NEXT: str w8, [sp], #16
; CHECK-NEXT: ret

attributes #0 = { noredzone "target-cpu"="cyclone" }
