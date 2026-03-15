; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 < %s | FileCheck %s

; CHECK-LABEL: __unnamed_1:
; CHECK: .set __unnamed_1.num_vgpr, 0
; CHECK: .set __unnamed_1.num_agpr, 0
; CHECK: .set __unnamed_1.numbered_sgpr, 32
; CHECK: .set __unnamed_1.private_seg_size, 0
; CHECK: .set __unnamed_1.uses_vcc, 0
; CHECK: .set __unnamed_1.uses_flat_scratch, 0
; CHECK: .set __unnamed_1.has_dyn_sized_stack, 0
; CHECK: .set __unnamed_1.has_recursion, 0
; CHECK: .set __unnamed_1.has_indirect_call, 0
define void @1() {
entry:
  ret void
}

; CHECK-LABEL: __unnamed_2:
; CHECK: .set __unnamed_2.num_vgpr, max(1, __unnamed_1.num_vgpr)
; CHECK: .set __unnamed_2.num_agpr, max(0, __unnamed_1.num_agpr)
; CHECK: .set __unnamed_2.numbered_sgpr, max(34, __unnamed_1.numbered_sgpr)
; CHECK: .set __unnamed_2.private_seg_size, 16+max(__unnamed_1.private_seg_size)
; CHECK: .set __unnamed_2.uses_vcc, or(0, __unnamed_1.uses_vcc)
; CHECK: .set __unnamed_2.uses_flat_scratch, or(0, __unnamed_1.uses_flat_scratch)
; CHECK: .set __unnamed_2.has_dyn_sized_stack, or(0, __unnamed_1.has_dyn_sized_stack)
; CHECK: .set __unnamed_2.has_recursion, or(1, __unnamed_1.has_recursion)
; CHECK: .set __unnamed_2.has_indirect_call, or(0, __unnamed_1.has_indirect_call)
define void @2() {
entry:
  call void @1()
  ret void
}

; CHECK-LABEL: {{^}}use
; CHECK: .set use.num_vgpr, max(32, __unnamed_1.num_vgpr, __unnamed_2.num_vgpr)
; CHECK: .set use.num_agpr, max(0, __unnamed_1.num_agpr, __unnamed_2.num_agpr)
; CHECK: .set use.numbered_sgpr, max(33, __unnamed_1.numbered_sgpr, __unnamed_2.numbered_sgpr)
; CHECK: .set use.private_seg_size, 0+max(__unnamed_1.private_seg_size, __unnamed_2.private_seg_size)
; CHECK: .set use.uses_vcc, or(0, __unnamed_1.uses_vcc, __unnamed_2.uses_vcc)
; CHECK: .set use.uses_flat_scratch, or(1, __unnamed_1.uses_flat_scratch, __unnamed_2.uses_flat_scratch)
; CHECK: .set use.has_dyn_sized_stack, or(0, __unnamed_1.has_dyn_sized_stack, __unnamed_2.has_dyn_sized_stack)
; CHECK: .set use.has_recursion, or(1, __unnamed_1.has_recursion, __unnamed_2.has_recursion)
; CHECK: .set use.has_indirect_call, or(0, __unnamed_1.has_indirect_call, __unnamed_2.has_indirect_call)
define amdgpu_kernel void @use() {
  call void @1()
  call void @2()
  ret void
}
