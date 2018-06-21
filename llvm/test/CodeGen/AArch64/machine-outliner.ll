; RUN: llc -enable-machine-outliner -mtriple=aarch64-apple-darwin < %s | FileCheck %s -check-prefix=NoODR
; RUN: llc -enable-machine-outliner -enable-linkonceodr-outlining -mtriple=aarch64-apple-darwin < %s | FileCheck %s -check-prefix=ODR

define linkonce_odr void @fish() #0 {
  ; CHECK-LABEL: _fish:
  ; NoODR:      orr w8, wzr, #0x1
  ; NoODR-NEXT: stp w8, wzr, [sp, #8]
  ; NoODR-NEXT: orr w8, wzr, #0x2
  ; NoODR-NEXT: str w8, [sp, #4]
  ; NoODR-NEXT: orr w8, wzr, #0x3
  ; NoODR-NEXT: str w8, [sp], #16
  ; NoODR-NEXT: ret
  ; ODR: b l_OUTLINED_FUNCTION_0
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
  ; CHECK: b l_OUTLINED_FUNCTION_0
  ; CHECK-NOT: ret
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
  ; CHECK: b l_OUTLINED_FUNCTION_0
  ; CHECK-NOT: ret
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

; CHECK-LABEL: l_OUTLINED_FUNCTION_0:
; CHECK:      orr w8, wzr, #0x1
; CHECK-NEXT: stp w8, wzr, [sp, #8]
; CHECK-NEXT: orr w8, wzr, #0x2
; CHECK-NEXT: str w8, [sp, #4]
; CHECK-NEXT: orr w8, wzr, #0x3
; CHECK-NEXT: str w8, [sp], #16
; CHECK-NEXT: ret

attributes #0 = { noredzone nounwind ssp uwtable "no-frame-pointer-elim"="false" "target-cpu"="cyclone" }
