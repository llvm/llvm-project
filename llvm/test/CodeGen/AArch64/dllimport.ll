; RUN: llc -mtriple aarch64-unknown-windows-msvc -filetype asm -o - %s | FileCheck %s

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
; CHECK: ldr w0, [x8, ext]
; CHECK: ret

define i32* @get_var_pointer() {
  ret i32* @var
}

; CHECK-LABEL: get_var_pointer
; CHECK: adrp x0, __imp_var
; CHECK: ldr x0, [x0, __imp_var]
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
; CHECK: b internal
