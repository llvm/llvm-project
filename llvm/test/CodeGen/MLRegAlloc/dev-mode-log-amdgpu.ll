; REQUIRES: have_tflite
; REQUIRES: x86_64-linux
; REQUIRES: amdgpu-registered-target
;
; AMDGPU runs the register allocator more than once per function. Check that
; each function gets a single context record.
;
; RUN: llc -o /dev/null -mtriple=amdgpu9.0a-amd-amdhsa \
; RUN:   -regalloc-enable-advisor=development -mlregalloc-num-allocatable-regs=256 \
; RUN:   -regalloc-training-log=%t1 < %s
; RUN: FileCheck --input-file %t1 %s

; RUN: llc -o /dev/null -mtriple=amdgpu9.0a-amd-amdhsa \
; RUN:   -regalloc-enable-priority-advisor=development \
; RUN:   -regalloc-priority-training-log=%t2 < %s
; RUN: FileCheck --input-file %t2 %s

; CHECK:     {"context":"f"}
; CHECK-NOT: {"context"
; CHECK:     {"context":"g"}
; CHECK-NOT: {"context"

define amdgpu_kernel void @f(ptr addrspace(1) %p) {
  %v = load volatile i32, ptr addrspace(1) %p
  store volatile i32 %v, ptr addrspace(1) %p
  ret void
}

define amdgpu_kernel void @g(ptr addrspace(1) %p) {
  %v = load volatile i32, ptr addrspace(1) %p
  store volatile i32 %v, ptr addrspace(1) %p
  ret void
}
