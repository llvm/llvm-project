; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 < %s | FileCheck %s

; CHECK-LABEL: __unnamed_1:
; CHECK: .set .L__unnamed_1.num_vgpr, 0
; CHECK: .set .L__unnamed_1.num_agpr, 0
; CHECK: .set .L__unnamed_1.numbered_sgpr, 32
; CHECK: .set .L__unnamed_1.private_seg_size, 0
; CHECK: .set .L__unnamed_1.uses_vcc, 0
; CHECK: .set .L__unnamed_1.uses_flat_scratch, 0
; CHECK: .set .L__unnamed_1.has_dyn_sized_stack, 0
; CHECK: .set .L__unnamed_1.has_recursion, 0
; CHECK: .set .L__unnamed_1.has_indirect_call, 0
define void @1() {
entry:
  ret void
}

; CHECK-LABEL: __unnamed_2:
; CHECK: .set .L__unnamed_2.num_vgpr, max(1, .L__unnamed_1.num_vgpr)
; CHECK: .set .L__unnamed_2.num_agpr, max(0, .L__unnamed_1.num_agpr)
; CHECK: .set .L__unnamed_2.numbered_sgpr, max(34, .L__unnamed_1.numbered_sgpr)
; CHECK: .set .L__unnamed_2.private_seg_size, 16+max(.L__unnamed_1.private_seg_size)
; CHECK: .set .L__unnamed_2.uses_vcc, or(0, .L__unnamed_1.uses_vcc)
; CHECK: .set .L__unnamed_2.uses_flat_scratch, or(0, .L__unnamed_1.uses_flat_scratch)
; CHECK: .set .L__unnamed_2.has_dyn_sized_stack, or(0, .L__unnamed_1.has_dyn_sized_stack)
; CHECK: .set .L__unnamed_2.has_recursion, or(1, .L__unnamed_1.has_recursion)
; CHECK: .set .L__unnamed_2.has_indirect_call, or(0, .L__unnamed_1.has_indirect_call)
define void @2() {
entry:
  call void @1()
  ret void
}

; CHECK-LABEL: {{^}}use
; CHECK: .set .Luse.num_vgpr, max(32, .L__unnamed_1.num_vgpr, .L__unnamed_2.num_vgpr)
; CHECK: .set .Luse.num_agpr, max(0, .L__unnamed_1.num_agpr, .L__unnamed_2.num_agpr)
; CHECK: .set .Luse.numbered_sgpr, max(33, .L__unnamed_1.numbered_sgpr, .L__unnamed_2.numbered_sgpr)
; CHECK: .set .Luse.private_seg_size, 0+max(.L__unnamed_1.private_seg_size, .L__unnamed_2.private_seg_size)
; CHECK: .set .Luse.uses_vcc, or(0, .L__unnamed_1.uses_vcc, .L__unnamed_2.uses_vcc)
; CHECK: .set .Luse.uses_flat_scratch, or(1, .L__unnamed_1.uses_flat_scratch, .L__unnamed_2.uses_flat_scratch)
; CHECK: .set .Luse.has_dyn_sized_stack, or(0, .L__unnamed_1.has_dyn_sized_stack, .L__unnamed_2.has_dyn_sized_stack)
; CHECK: .set .Luse.has_recursion, or(1, .L__unnamed_1.has_recursion, .L__unnamed_2.has_recursion)
; CHECK: .set .Luse.has_indirect_call, or(0, .L__unnamed_1.has_indirect_call, .L__unnamed_2.has_indirect_call)
define amdgpu_kernel void @use() {
  call void @1()
  call void @2()
  ret void
}
