; Address-space 2 (constant buffer) pointer parameters are packed into the
; trailing `_scalar_buf` device buffer alongside scalar params — i.e. the
; AS(2) input parameter disappears from the post-pipeline signature.
; RUN: llc -mtriple=air %s -o - | FileCheck %s

; CHECK-LABEL: define void @kernel
; CHECK-SAME: ptr addrspace(1){{.*}} %out, ptr addrspace(1){{.*}} %_scalar_buf
; CHECK-NOT: ptr addrspace(2)
; CHECK: ret void

define void @kernel(ptr addrspace(1) %out, ptr addrspace(2) %cbuf, i32 %n) {
entry:
  %v = load i32, ptr addrspace(2) %cbuf, align 4
  %s = add i32 %v, %n
  store i32 %s, ptr addrspace(1) %out, align 4
  ret void
}
