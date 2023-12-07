; RUN: llc -mtriple thumbv7-windows -filetype asm -o - %s | FileCheck %s

; ModuleID = 'dllimport.c'

@var = external dllimport global i32
@ext = external global i32
declare dllimport arm_aapcs_vfpcc i32 @external()
declare arm_aapcs_vfpcc i32 @internal()

define arm_aapcs_vfpcc i32 @get_var() {
  %1 = load i32, ptr @var, align 4
  ret i32 %1
}

; CHECK-LABEL: get_var
; CHECK: movw r0, :lower16:__imp_var
; CHECK: movt r0, :upper16:__imp_var
; CHECK: ldr r0, [r0]
; CHECK: ldr r0, [r0]
; CHECK: bx lr

define arm_aapcs_vfpcc i32 @get_ext() {
  %1 = load i32, ptr @ext, align 4
  ret i32 %1
}

; CHECK-LABEL: get_ext
; CHECK: movw r0, :lower16:ext
; CHECK: movt r0, :upper16:ext
; CHECK: ldr r0, [r0]
; CHECK: bx lr

define arm_aapcs_vfpcc ptr @get_var_pointer() {
  ret ptr @var
}

; CHECK-LABEL: get_var_pointer
; CHECK:  movw r0, :lower16:__imp_var
; CHECK:  movt r0, :upper16:__imp_var
; CHECK:  ldr r0, [r0]
; CHECK:  bx lr

define arm_aapcs_vfpcc i32 @call_external() {
  %call = tail call arm_aapcs_vfpcc i32 @external()
  ret i32 %call
}

; CHECK-LABEL: call_external
; CHECK: movw r0, :lower16:__imp_external
; CHECK: movt r0, :upper16:__imp_external
; CHECK: ldr r0, [r0]
; CHECK: bx r0

define arm_aapcs_vfpcc i32 @call_internal() {
  %call = tail call arm_aapcs_vfpcc i32 @internal()
  ret i32 %call
}

; CHECK-LABEL: call_internal
; CHECK: b.w internal

