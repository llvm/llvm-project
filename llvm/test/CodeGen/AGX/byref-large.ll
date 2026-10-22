; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s

; By-reference arguments are passed in uniforms as long as they fit in the
; 1 KiB uniform file with the other arguments and the implicit ones
; (ptx-re-blockers B9: all 512 halves preload): a 600-byte struct is read from
; uniforms, a 1,100-byte one stays a pointer and is read from memory.

; CHECK-LABEL: fits:
; CHECK: // uniform registers: {{3[0-9][0-9]}} x 16-bit
; CHECK-NOT: device_load
; CHECK: device_store
define agx_kernel void @fits(ptr addrspace(1) %out, ptr addrspace(4) byref([150 x i32]) align 4 %s) {
  %p = getelementptr inbounds i8, ptr addrspace(4) %s, i64 596
  %v = load i32, ptr addrspace(4) %p, align 4
  store i32 %v, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: too_large:
; CHECK: // uniform registers: {{[0-9]+}} x 16-bit
; CHECK: device_load
define agx_kernel void @too_large(ptr addrspace(1) %out, ptr addrspace(4) byref([275 x i32]) align 4 %s) {
  %p = getelementptr inbounds i8, ptr addrspace(4) %s, i64 1096
  %v = load i32, ptr addrspace(4) %p, align 4
  store i32 %v, ptr addrspace(1) %out, align 4
  ret void
}
