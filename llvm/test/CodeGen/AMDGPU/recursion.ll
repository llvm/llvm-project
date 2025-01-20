; RUN: sed 's/CODE_OBJECT_VERSION/400/g' %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -verify-machineinstrs | FileCheck %s
; RUN: sed 's/CODE_OBJECT_VERSION/500/g' %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -verify-machineinstrs | FileCheck -check-prefixes=V5 %s
; RUN: sed 's/CODE_OBJECT_VERSION/600/g' %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -verify-machineinstrs | FileCheck -check-prefixes=V5 %s

; CHECK-LABEL: {{^}}recursive:
; CHECK: .set recursive.private_seg_size, 16+(max(16384))
; CHECK: ScratchSize: 16

; V5-LABEL: {{^}}recursive:
; V5: .set recursive.has_recursion, 1
define void @recursive() {
  call void @recursive()
  store volatile i32 0, ptr addrspace(1) undef
  ret void
}

; CHECK-LABEL: {{^}}tail_recursive:
; CHECK: .set tail_recursive.private_seg_size, 0
; CHECK: ScratchSize: 0
define void @tail_recursive() {
  tail call void @tail_recursive()
  ret void
}

; CHECK: .set calls_tail_recursive.private_seg_size, 0+(max(tail_recursive.private_seg_size))
define void @calls_tail_recursive() norecurse {
  tail call void @tail_recursive()
  ret void
}

; CHECK-LABEL: {{^}}tail_recursive_with_stack:
; CHECK: .set tail_recursive_with_stack.private_seg_size, 8
; CHECK: .set tail_recursive_with_stack.has_recursion, 1
define void @tail_recursive_with_stack() {
  %alloca = alloca i32, addrspace(5)
  store volatile i32 0, ptr addrspace(5) %alloca
  tail call void @tail_recursive_with_stack()
  ret void
}

; For an arbitrary recursive call, report a large number for unknown stack
; usage for code object v4 and older
; CHECK-LABEL: {{^}}calls_recursive:
; CHECK: .set calls_recursive.private_seg_size, 0+(max(16384, recursive.private_seg_size))
;
; V5-LABEL: {{^}}calls_recursive:
; V5: .set calls_recursive.private_seg_size, 0+(max(recursive.private_seg_size))
; V5: .set calls_recursive.has_dyn_sized_stack, or(0, recursive.has_dyn_sized_stack)
define amdgpu_kernel void @calls_recursive() {
  call void @recursive()
  ret void
}

; Make sure we do not report a huge stack size for tail recursive
; functions
; CHECK-LABEL: {{^}}kernel_indirectly_calls_tail_recursive:
; CHECK: .set kernel_indirectly_calls_tail_recursive.private_seg_size, 0+(max(calls_tail_recursive.private_seg_size))
define amdgpu_kernel void @kernel_indirectly_calls_tail_recursive() {
  call void @calls_tail_recursive()
  ret void
}

; TODO: Even though tail_recursive is only called as a tail call, we
; end up treating it as generally recursive call from the regular call
; in the kernel.

; CHECK-LABEL: {{^}}kernel_calls_tail_recursive:
; CHECK: .set kernel_calls_tail_recursive.private_seg_size, 0+(max(16384, tail_recursive.private_seg_size))
;
; V5-LABEL: {{^}}kernel_calls_tail_recursive:
; V5: .set kernel_calls_tail_recursive.private_seg_size, 0+(max(tail_recursive.private_seg_size))
; V5: .set kernel_calls_tail_recursive.has_recursion, or(1, tail_recursive.has_recursion)
define amdgpu_kernel void @kernel_calls_tail_recursive() {
  call void @tail_recursive()
  ret void
}

; CHECK-LABEL: {{^}}kernel_calls_tail_recursive_with_stack:
; CHECK: .set kernel_calls_tail_recursive_with_stack.private_seg_size, 0+(max(16384, tail_recursive_with_stack.private_seg_size))
;
; V5-LABEL: {{^}}kernel_calls_tail_recursive_with_stack:
; V5: .set kernel_calls_tail_recursive_with_stack.private_seg_size, 0+(max(tail_recursive_with_stack.private_seg_size))
; V5: .set kernel_calls_tail_recursive_with_stack.has_dyn_sized_stack, or(0, tail_recursive_with_stack.has_dyn_sized_stack)
define amdgpu_kernel void @kernel_calls_tail_recursive_with_stack() {
  call void @tail_recursive_with_stack()
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 CODE_OBJECT_VERSION}
