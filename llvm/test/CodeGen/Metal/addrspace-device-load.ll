; Address-space 1 (device buffer) loads/stores survive the pipeline with their
; addrspace annotation intact. The scalar-buffer-packing pass moves the
; addrspace(1) param to the front of the post-packing kernel signature.
; RUN: llc -mtriple=air %s -o - | FileCheck %s

; CHECK-LABEL: define void @kernel
; CHECK-SAME: ptr addrspace(1){{.*}} %p
; CHECK: load i32, ptr addrspace(1)
; CHECK: store i32 {{.*}}, ptr addrspace(1)
; CHECK: ret void

define void @kernel(ptr addrspace(1) %p) {
entry:
  %v = load i32, ptr addrspace(1) %p, align 4
  %s = add i32 %v, 1
  store i32 %s, ptr addrspace(1) %p, align 4
  ret void
}
