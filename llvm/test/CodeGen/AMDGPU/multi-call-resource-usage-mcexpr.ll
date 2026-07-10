; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a < %s | FileCheck %s

; CHECK-LABEL: {{^}}qux
; CHECK: .set .Lqux.num_vgpr, 13
; CHECK: .set .Lqux.num_agpr, 0
; CHECK: .set .Lqux.numbered_sgpr, 32
; CHECK: .set .Lqux.private_seg_size, 0
; CHECK: .set .Lqux.uses_vcc, 0
; CHECK: .set .Lqux.uses_flat_scratch, 0
; CHECK: .set .Lqux.has_dyn_sized_stack, 0
; CHECK: .set .Lqux.has_recursion, 0
; CHECK: .set .Lqux.has_indirect_call, 0
define void @qux() {
entry:
  call void asm sideeffect "", "~{v12}"()
  ret void
}

; CHECK-LABEL: {{^}}baz
; CHECK: .set .Lbaz.num_vgpr, max(49, .Lqux.num_vgpr)
; CHECK: .set .Lbaz.num_agpr, max(0, .Lqux.num_agpr)
; CHECK: .set .Lbaz.numbered_sgpr, max(34, .Lqux.numbered_sgpr)
; CHECK: .set .Lbaz.private_seg_size, 16+max(.Lqux.private_seg_size)
; CHECK: .set .Lbaz.uses_vcc, or(0, .Lqux.uses_vcc)
; CHECK: .set .Lbaz.uses_flat_scratch, or(0, .Lqux.uses_flat_scratch)
; CHECK: .set .Lbaz.has_dyn_sized_stack, or(0, .Lqux.has_dyn_sized_stack)
; CHECK: .set .Lbaz.has_recursion, or(1, .Lqux.has_recursion)
; CHECK: .set .Lbaz.has_indirect_call, or(0, .Lqux.has_indirect_call)
define void @baz() {
entry:
  call void @qux()
  call void asm sideeffect "", "~{v48}"()
  ret void
}

; CHECK-LABEL: {{^}}bar
; CHECK: .set .Lbar.num_vgpr, max(65, .Lbaz.num_vgpr, .Lqux.num_vgpr)
; CHECK: .set .Lbar.num_agpr, max(0, .Lbaz.num_agpr, .Lqux.num_agpr)
; CHECK: .set .Lbar.numbered_sgpr, max(34, .Lbaz.numbered_sgpr, .Lqux.numbered_sgpr)
; CHECK: .set .Lbar.private_seg_size, 16+max(.Lbaz.private_seg_size, .Lqux.private_seg_size)
; CHECK: .set .Lbar.uses_vcc, or(0, .Lbaz.uses_vcc, .Lqux.uses_vcc)
; CHECK: .set .Lbar.uses_flat_scratch, or(0, .Lbaz.uses_flat_scratch, .Lqux.uses_flat_scratch)
; CHECK: .set .Lbar.has_dyn_sized_stack, or(0, .Lbaz.has_dyn_sized_stack, .Lqux.has_dyn_sized_stack)
; CHECK: .set .Lbar.has_recursion, or(1, .Lbaz.has_recursion, .Lqux.has_recursion)
; CHECK: .set .Lbar.has_indirect_call, or(0, .Lbaz.has_indirect_call, .Lqux.has_indirect_call)
define void @bar() {
entry:
  call void @baz()
  call void @qux()
  call void @baz()
  call void asm sideeffect "", "~{v64}"()
  ret void
}

; CHECK-LABEL: {{^}}foo
; CHECK: .set .Lfoo.num_vgpr, max(38, .Lbar.num_vgpr)
; CHECK: .set .Lfoo.num_agpr, max(0, .Lbar.num_agpr)
; CHECK: .set .Lfoo.numbered_sgpr, max(34, .Lbar.numbered_sgpr)
; CHECK: .set .Lfoo.private_seg_size, 16+max(.Lbar.private_seg_size)
; CHECK: .set .Lfoo.uses_vcc, or(0, .Lbar.uses_vcc)
; CHECK: .set .Lfoo.uses_flat_scratch, or(0, .Lbar.uses_flat_scratch)
; CHECK: .set .Lfoo.has_dyn_sized_stack, or(0, .Lbar.has_dyn_sized_stack)
; CHECK: .set .Lfoo.has_recursion, or(1, .Lbar.has_recursion)
; CHECK: .set .Lfoo.has_indirect_call, or(0, .Lbar.has_indirect_call)
define void @foo() {
entry:
  call void @bar()
  call void asm sideeffect "", "~{v37}"()
  ret void
}

; CHECK-LABEL: {{^}}usefoo
; CHECK: .set .Lusefoo.num_vgpr, max(32, .Lfoo.num_vgpr)
; CHECK: .set .Lusefoo.num_agpr, max(0, .Lfoo.num_agpr)
; CHECK: .set .Lusefoo.numbered_sgpr, max(33, .Lfoo.numbered_sgpr)
; CHECK: .set .Lusefoo.private_seg_size, 0+max(.Lfoo.private_seg_size)
; CHECK: .set .Lusefoo.uses_vcc, or(0, .Lfoo.uses_vcc)
; CHECK: .set .Lusefoo.uses_flat_scratch, or(1, .Lfoo.uses_flat_scratch)
; CHECK: .set .Lusefoo.has_dyn_sized_stack, or(0, .Lfoo.has_dyn_sized_stack)
; CHECK: .set .Lusefoo.has_recursion, or(1, .Lfoo.has_recursion)
; CHECK: .set .Lusefoo.has_indirect_call, or(0, .Lfoo.has_indirect_call)
define amdgpu_kernel void @usefoo() {
  call void @foo()
  ret void
}

