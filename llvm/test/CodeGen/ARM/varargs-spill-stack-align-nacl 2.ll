; RUN: llc < %s -mtriple=arm-nacl-gnueabi | FileCheck %s

declare void @llvm.va_start(ptr)
declare void @external_func(ptr)

@va_list = external global ptr

; On ARM, varargs arguments are passed in r0-r3 with the rest on the
; stack.  A varargs function must therefore spill rN-r3 just below the
; function's initial stack pointer.
;
; This test checks for a bug in which a gap was left between the spill
; area and varargs arguments on the stack when using 16 byte stack
; alignment.

define void @varargs_func(i32 %arg1, ...) {
  call void @llvm.va_start(ptr @va_list)
  call void @external_func(ptr @va_list)
  ret void
}
; CHECK-LABEL: varargs_func:
; Reserve space for the varargs save area.  This currently reserves
; more than enough (16 bytes rather than the 12 bytes needed).
; CHECK: sub sp, sp, #12
; CHECK: push {r11, lr}
; Align the stack pointer to a multiple of 16.
; CHECK: sub sp, sp, #12
; Calculate the address of the varargs save area and save varargs
; arguments into it.
; CHECK-NEXT: add r0, sp, #20
; CHECK-NEXT: stm r0, {r1, r2, r3}
