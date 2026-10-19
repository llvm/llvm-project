; RUN: llc -mtriple=agx -mcpu=agx-g13g -verify-machineinstrs < %s | FileCheck %s

; Functions other than kernels are inlined and removed, used or not.

; CHECK-NOT: helper
; CHECK-NOT: unused
; CHECK-LABEL: kernel:
; CHECK: iadd {{r[0-9]+}}, u2, 5
; CHECK-NOT: helper
; CHECK-NOT: unused

define i32 @helper(i32 %x) {
  %r = add i32 %x, 5
  ret i32 %r
}

define hidden float @unused(float %x) {
  ret float %x
}

define agx_kernel void @kernel(ptr addrspace(1) %out, i32 %x) {
  %r = call i32 @helper(i32 %x)
  store i32 %r, ptr addrspace(1) %out
  ret void
}
