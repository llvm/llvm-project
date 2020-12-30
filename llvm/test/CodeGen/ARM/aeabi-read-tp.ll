; RUN: llc -mtriple armv7---eabi -filetype asm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-SHORT
; RUN: llc -mtriple thumbv7---eabi -filetype asm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-SHORT
; RUN: llc -mtriple armv7---eabi -mattr=+long-calls -filetype asm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-LONG
; RUN: llc -mtriple thumbv7---eabi -mattr=+long-calls -filetype asm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-LONG

@i = dso_local thread_local local_unnamed_addr global i32 0, align 4

define dso_local i32 @f() local_unnamed_addr {
entry:
  %0 = load i32, i32* @i, align 4
  ret i32 %0
}

; CHECK-LABEL: f:
; CHECK-SHORT: ldr r1, [[VAR:.LCPI[0-9]+_[0-9]+]]
; CHECK-SHORT-NEXT: bl __aeabi_read_tp
; CHECK-SHORT: [[VAR]]:
; CHECK-SHORT-NEXT: .long i(TPOFF)

; CHECK-LONG: ldr [[REG:r[0-9]+]], [[FUN:.LCPI[0-9]+_[0-9]+]]
; CHECK-LONG-NEXT: ldr r1, [[VAR:.LCPI[0-9]+_[0-9]+]]
; CHECK-LONG-NEXT: blx [[REG]]
; CHECK-LONG: [[VAR]]:
; CHECK-LONG-NEXT: .long i(TPOFF)
; CHECK-LONG: [[FUN]]:
; CHECK-LONG-NEXT: .long __aeabi_read_tp
