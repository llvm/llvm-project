; RUN: llc -mtriple aarch64-unknown-windows-msvc -filetype asm -o - %s | FileCheck %s -check-prefixes=CHECK,DAG-ISEL
; RUN: llc -mtriple aarch64-unknown-windows-msvc -fast-isel -filetype asm -o - %s | FileCheck %s -check-prefixes=CHECK,FAST-ISEL
; RUN: llc -mtriple aarch64-unknown-windows-msvc -verify-machineinstrs -O0 -filetype asm -o - %s | FileCheck %s -check-prefixes=CHECK,GLOBAL-ISEL,GLOBAL-ISEL-FALLBACK

@var = external dllimport global i32
@ext = external global i32
declare dllimport i32 @external()
declare i32 @internal()

define i32 @get_var() {
  %1 = load i32, i32* @var, align 4
  ret i32 %1
}

; CHECK-LABEL: get_var
; CHECK: adrp x8, __imp_var
; CHECK: ldr x8, [x8, __imp_var]
; CHECK: ldr w0, [x8]
; CHECK: ret

define i32 @get_ext() {
  %1 = load i32, i32* @ext, align 4
  ret i32 %1
}

; CHECK-LABEL: get_ext
; CHECK: adrp x8, ext
; DAG-ISEL: ldr w0, [x8, ext]
; FAST-ISEL: add x8, x8, ext
; FAST-ISEL: ldr w0, [x8]
; GLOBAL-ISEL-FALLBACK: add x8, x8, ext
; GLOBAL-ISEL-FALLBACK: ldr w0, [x8]
; CHECK: ret

define i32* @get_var_pointer() {
  ret i32* @var
}

; CHECK-LABEL: get_var_pointer
; CHECK: adrp [[REG1:x[0-9]+]], __imp_var
; CHECK: ldr {{x[0-9]+}}, {{\[}}[[REG1]], __imp_var]
; CHECK: ret

define i32 @call_external() {
  %call = tail call i32 @external()
  ret i32 %call
}

; CHECK-LABEL: call_external
; CHECK: adrp x0, __imp_external
; CHECK: ldr x0, [x0, __imp_external]
; CHECK: br x0

define i32 @call_internal() {
  %call = tail call i32 @internal()
  ret i32 %call
}

; CHECK-LABEL: call_internal
; DAG-ISEL: b internal
; FAST-ISEL: b internal
; GLOBAL-ISEL: b internal
