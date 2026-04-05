; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu | FileCheck %s
; RUN: llc -verify-machineinstrs -global-isel < %s -mtriple=aarch64-none-linux-gnu | FileCheck %s

; Check the SPIR call conventions work.

define spir_func i32 @callee(i32 %a, i32 %b) {
; CHECK-LABEL: callee:
; CHECK: add w0, w0, w1
; CHECK: ret
  %sum = add i32 %a, %b
  ret i32 %sum
}

define spir_func i32 @caller(i32 %x) {
; CHECK-LABEL: caller:
; CHECK: mov w1, #7
; CHECK: bl callee
; CHECK: ret
  %call = call spir_func i32 @callee(i32 %x, i32 7)
  ret i32 %call
}

declare spir_func i64 @_Z13get_global_idj(i32)

define spir_kernel void @addVectors(ptr %a, ptr %b, ptr %c) {
; CHECK-LABEL: addVectors:
; CHECK: bl _Z13get_global_idj
; CHECK: ret
  %gid = call spir_func i64 @_Z13get_global_idj(i32 0)
  ret void
}
