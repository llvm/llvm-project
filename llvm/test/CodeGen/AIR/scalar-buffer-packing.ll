; Scalar kernel parameters and `addrspace(2)` constant-buffer pointers are
; packed into a single trailing `addrspace(1)` device-buffer parameter named
; `_scalar_buf`, with a per-field GEP+load preamble inserted at function entry.
; RUN: llc -mtriple=air %s -o - | FileCheck %s

; CHECK-LABEL: define void @kernel
; The original i32 / float scalar params are gone; the device buffer stays
; first and a packed scalar buffer is appended at the end (followed by any
; AIR system value params injected by AIRSystemValues).
; CHECK-SAME: ptr addrspace(1){{.*}} %out, ptr addrspace(1){{.*}} %_scalar_buf
; CHECK: getelementptr inbounds float, ptr addrspace(1) %_scalar_buf, i64 0
; CHECK: getelementptr inbounds float, ptr addrspace(1) %_scalar_buf, i64 1
; CHECK: store

define void @kernel(ptr addrspace(1) %out, i32 %n, float %scale) {
entry:
  %sx = sitofp i32 %n to float
  %sum = fadd float %sx, %scale
  store float %sum, ptr addrspace(1) %out
  ret void
}
