; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -enable-ipra=0 < %s | FileCheck -check-prefix=GCN %s

; Functions that don't make calls should have constants as its resource usage as no resource information has to be propagated.

; GCN-LABEL: {{^}}use_vcc:
; GCN: .set use_vcc.num_vgpr, 0
; GCN: .set use_vcc.num_agpr, 0
; GCN: .set use_vcc.numbered_sgpr, 32
; GCN: .set use_vcc.private_seg_size, 0
; GCN: .set use_vcc.uses_vcc, 1
; GCN: .set use_vcc.uses_flat_scratch, 0
; GCN: .set use_vcc.has_dyn_sized_stack, 0
; GCN: .set use_vcc.has_recursion, 0
; GCN: .set use_vcc.has_indirect_call, 0
; GCN: TotalNumSgprs: 36
; GCN: NumVgprs: 0
; GCN: ScratchSize: 0
define void @use_vcc() #1 {
  call void asm sideeffect "", "~{vcc}" () #0
  ret void
}

; GCN-LABEL: {{^}}indirect_use_vcc:
; GCN: .set indirect_use_vcc.num_vgpr, max(41, use_vcc.num_vgpr)
; GCN: .set indirect_use_vcc.num_agpr, max(0, use_vcc.num_agpr)
; GCN: .set indirect_use_vcc.numbered_sgpr, max(34, use_vcc.numbered_sgpr)
; GCN: .set indirect_use_vcc.private_seg_size, 16+(max(use_vcc.private_seg_size))
; GCN: .set indirect_use_vcc.uses_vcc, or(1, use_vcc.uses_vcc)
; GCN: .set indirect_use_vcc.uses_flat_scratch, or(0, use_vcc.uses_flat_scratch)
; GCN: .set indirect_use_vcc.has_dyn_sized_stack, or(0, use_vcc.has_dyn_sized_stack)
; GCN: .set indirect_use_vcc.has_recursion, or(0, use_vcc.has_recursion)
; GCN: .set indirect_use_vcc.has_indirect_call, or(0, use_vcc.has_indirect_call)
; GCN: TotalNumSgprs: 38
; GCN: NumVgprs: 41
; GCN: ScratchSize: 16
define void @indirect_use_vcc() #1 {
  call void @use_vcc()
  ret void
}

; GCN-LABEL: {{^}}indirect_2level_use_vcc_kernel:
; GCN: .set indirect_2level_use_vcc_kernel.num_vgpr, max(32, indirect_use_vcc.num_vgpr)
; GCN: .set indirect_2level_use_vcc_kernel.num_agpr, max(0, indirect_use_vcc.num_agpr)
; GCN: .set indirect_2level_use_vcc_kernel.numbered_sgpr, max(33, indirect_use_vcc.numbered_sgpr)
; GCN: .set indirect_2level_use_vcc_kernel.private_seg_size, 0+(max(indirect_use_vcc.private_seg_size))
; GCN: .set indirect_2level_use_vcc_kernel.uses_vcc, or(1, indirect_use_vcc.uses_vcc)
; GCN: .set indirect_2level_use_vcc_kernel.uses_flat_scratch, or(1, indirect_use_vcc.uses_flat_scratch)
; GCN: .set indirect_2level_use_vcc_kernel.has_dyn_sized_stack, or(0, indirect_use_vcc.has_dyn_sized_stack)
; GCN: .set indirect_2level_use_vcc_kernel.has_recursion, or(0, indirect_use_vcc.has_recursion)
; GCN: .set indirect_2level_use_vcc_kernel.has_indirect_call, or(0, indirect_use_vcc.has_indirect_call)
; GCN: TotalNumSgprs: 40
; GCN: NumVgprs: 41
; GCN: ScratchSize: 16
define amdgpu_kernel void @indirect_2level_use_vcc_kernel(ptr addrspace(1) %out) #0 {
  call void @indirect_use_vcc()
  ret void
}

; GCN-LABEL: {{^}}use_flat_scratch:
; GCN: .set use_flat_scratch.num_vgpr, 0
; GCN: .set use_flat_scratch.num_agpr, 0
; GCN: .set use_flat_scratch.numbered_sgpr, 32
; GCN: .set use_flat_scratch.private_seg_size, 0
; GCN: .set use_flat_scratch.uses_vcc, 0
; GCN: .set use_flat_scratch.uses_flat_scratch, 1
; GCN: .set use_flat_scratch.has_dyn_sized_stack, 0
; GCN: .set use_flat_scratch.has_recursion, 0
; GCN: .set use_flat_scratch.has_indirect_call, 0
; GCN: TotalNumSgprs: 38
; GCN: NumVgprs: 0
; GCN: ScratchSize: 0
define void @use_flat_scratch() #1 {
  call void asm sideeffect "", "~{flat_scratch}" () #0
  ret void
}

; GCN-LABEL: {{^}}indirect_use_flat_scratch:
; GCN: .set indirect_use_flat_scratch.num_vgpr, max(41, use_flat_scratch.num_vgpr)
; GCN: .set indirect_use_flat_scratch.num_agpr, max(0, use_flat_scratch.num_agpr)
; GCN: .set indirect_use_flat_scratch.numbered_sgpr, max(34, use_flat_scratch.numbered_sgpr)
; GCN: .set indirect_use_flat_scratch.private_seg_size, 16+(max(use_flat_scratch.private_seg_size))
; GCN: .set indirect_use_flat_scratch.uses_vcc, or(1, use_flat_scratch.uses_vcc)
; GCN: .set indirect_use_flat_scratch.uses_flat_scratch, or(0, use_flat_scratch.uses_flat_scratch)
; GCN: .set indirect_use_flat_scratch.has_dyn_sized_stack, or(0, use_flat_scratch.has_dyn_sized_stack)
; GCN: .set indirect_use_flat_scratch.has_recursion, or(0, use_flat_scratch.has_recursion)
; GCN: .set indirect_use_flat_scratch.has_indirect_call, or(0, use_flat_scratch.has_indirect_call)
; GCN: TotalNumSgprs: 40
; GCN: NumVgprs: 41
; GCN: ScratchSize: 16
define void @indirect_use_flat_scratch() #1 {
  call void @use_flat_scratch()
  ret void
}

; GCN-LABEL: {{^}}indirect_2level_use_flat_scratch_kernel:
; GCN: .set indirect_2level_use_flat_scratch_kernel.num_vgpr, max(32, indirect_use_flat_scratch.num_vgpr)
; GCN: .set indirect_2level_use_flat_scratch_kernel.num_agpr, max(0, indirect_use_flat_scratch.num_agpr)
; GCN: .set indirect_2level_use_flat_scratch_kernel.numbered_sgpr, max(33, indirect_use_flat_scratch.numbered_sgpr)
; GCN: .set indirect_2level_use_flat_scratch_kernel.private_seg_size, 0+(max(indirect_use_flat_scratch.private_seg_size))
; GCN: .set indirect_2level_use_flat_scratch_kernel.uses_vcc, or(1, indirect_use_flat_scratch.uses_vcc)
; GCN: .set indirect_2level_use_flat_scratch_kernel.uses_flat_scratch, or(1, indirect_use_flat_scratch.uses_flat_scratch)
; GCN: .set indirect_2level_use_flat_scratch_kernel.has_dyn_sized_stack, or(0, indirect_use_flat_scratch.has_dyn_sized_stack)
; GCN: .set indirect_2level_use_flat_scratch_kernel.has_recursion, or(0, indirect_use_flat_scratch.has_recursion)
; GCN: .set indirect_2level_use_flat_scratch_kernel.has_indirect_call, or(0, indirect_use_flat_scratch.has_indirect_call)
; GCN: TotalNumSgprs: 40
; GCN: NumVgprs: 41
; GCN: ScratchSize: 16
define amdgpu_kernel void @indirect_2level_use_flat_scratch_kernel(ptr addrspace(1) %out) #0 {
  call void @indirect_use_flat_scratch()
  ret void
}

; GCN-LABEL: {{^}}use_10_vgpr:
; GCN: .set use_10_vgpr.num_vgpr, 10
; GCN: .set use_10_vgpr.num_agpr, 0
; GCN: .set use_10_vgpr.numbered_sgpr, 32
; GCN: .set use_10_vgpr.private_seg_size, 0
; GCN: .set use_10_vgpr.uses_vcc, 0
; GCN: .set use_10_vgpr.uses_flat_scratch, 0
; GCN: .set use_10_vgpr.has_dyn_sized_stack, 0
; GCN: .set use_10_vgpr.has_recursion, 0
; GCN: .set use_10_vgpr.has_indirect_call, 0
; GCN: TotalNumSgprs: 36
; GCN: NumVgprs: 10
; GCN: ScratchSize: 0
define void @use_10_vgpr() #1 {
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4}"() #0
  call void asm sideeffect "", "~{v5},~{v6},~{v7},~{v8},~{v9}"() #0
  ret void
}

; GCN-LABEL: {{^}}indirect_use_10_vgpr:
; GCN: .set indirect_use_10_vgpr.num_vgpr, max(41, use_10_vgpr.num_vgpr)
; GCN: .set indirect_use_10_vgpr.num_agpr, max(0, use_10_vgpr.num_agpr)
; GCN: .set indirect_use_10_vgpr.numbered_sgpr, max(34, use_10_vgpr.numbered_sgpr)
; GCN: .set indirect_use_10_vgpr.private_seg_size, 16+(max(use_10_vgpr.private_seg_size))
; GCN: .set indirect_use_10_vgpr.uses_vcc, or(1, use_10_vgpr.uses_vcc)
; GCN: .set indirect_use_10_vgpr.uses_flat_scratch, or(0, use_10_vgpr.uses_flat_scratch)
; GCN: .set indirect_use_10_vgpr.has_dyn_sized_stack, or(0, use_10_vgpr.has_dyn_sized_stack)
; GCN: .set indirect_use_10_vgpr.has_recursion, or(0, use_10_vgpr.has_recursion)
; GCN: .set indirect_use_10_vgpr.has_indirect_call, or(0, use_10_vgpr.has_indirect_call)
; GCN: TotalNumSgprs: 38
; GCN: NumVgprs: 41
; GCN: ScratchSize: 16
define void @indirect_use_10_vgpr() #0 {
  call void @use_10_vgpr()
  ret void
}

; GCN-LABEL: {{^}}indirect_2_level_use_10_vgpr:
; GCN:	.set indirect_2_level_use_10_vgpr.num_vgpr, max(32, indirect_use_10_vgpr.num_vgpr)
; GCN:	.set indirect_2_level_use_10_vgpr.num_agpr, max(0, indirect_use_10_vgpr.num_agpr)
; GCN:	.set indirect_2_level_use_10_vgpr.numbered_sgpr, max(33, indirect_use_10_vgpr.numbered_sgpr)
; GCN:	.set indirect_2_level_use_10_vgpr.private_seg_size, 0+(max(indirect_use_10_vgpr.private_seg_size))
; GCN:	.set indirect_2_level_use_10_vgpr.uses_vcc, or(1, indirect_use_10_vgpr.uses_vcc)
; GCN:	.set indirect_2_level_use_10_vgpr.uses_flat_scratch, or(1, indirect_use_10_vgpr.uses_flat_scratch)
; GCN:	.set indirect_2_level_use_10_vgpr.has_dyn_sized_stack, or(0, indirect_use_10_vgpr.has_dyn_sized_stack)
; GCN:	.set indirect_2_level_use_10_vgpr.has_recursion, or(0, indirect_use_10_vgpr.has_recursion)
; GCN:	.set indirect_2_level_use_10_vgpr.has_indirect_call, or(0, indirect_use_10_vgpr.has_indirect_call)
; GCN: TotalNumSgprs: 40
; GCN: NumVgprs: 41
; GCN: ScratchSize: 16
define amdgpu_kernel void @indirect_2_level_use_10_vgpr() #0 {
  call void @indirect_use_10_vgpr()
  ret void
}

; GCN-LABEL: {{^}}use_50_vgpr:
; GCN:	.set use_50_vgpr.num_vgpr, 50
; GCN:	.set use_50_vgpr.num_agpr, 0
; GCN:	.set use_50_vgpr.numbered_sgpr, 32
; GCN:	.set use_50_vgpr.private_seg_size, 0
; GCN:	.set use_50_vgpr.uses_vcc, 0
; GCN:	.set use_50_vgpr.uses_flat_scratch, 0
; GCN:	.set use_50_vgpr.has_dyn_sized_stack, 0
; GCN:	.set use_50_vgpr.has_recursion, 0
; GCN:	.set use_50_vgpr.has_indirect_call, 0
; GCN: TotalNumSgprs: 36
; GCN: NumVgprs: 50
; GCN: ScratchSize: 0
define void @use_50_vgpr() #1 {
  call void asm sideeffect "", "~{v49}"() #0
  ret void
}

; GCN-LABEL: {{^}}indirect_use_50_vgpr:
; GCN:	.set indirect_use_50_vgpr.num_vgpr, max(41, use_50_vgpr.num_vgpr)
; GCN:	.set indirect_use_50_vgpr.num_agpr, max(0, use_50_vgpr.num_agpr)
; GCN:	.set indirect_use_50_vgpr.numbered_sgpr, max(34, use_50_vgpr.numbered_sgpr)
; GCN:	.set indirect_use_50_vgpr.private_seg_size, 16+(max(use_50_vgpr.private_seg_size))
; GCN:	.set indirect_use_50_vgpr.uses_vcc, or(1, use_50_vgpr.uses_vcc)
; GCN:	.set indirect_use_50_vgpr.uses_flat_scratch, or(0, use_50_vgpr.uses_flat_scratch)
; GCN:	.set indirect_use_50_vgpr.has_dyn_sized_stack, or(0, use_50_vgpr.has_dyn_sized_stack)
; GCN:	.set indirect_use_50_vgpr.has_recursion, or(0, use_50_vgpr.has_recursion)
; GCN:	.set indirect_use_50_vgpr.has_indirect_call, or(0, use_50_vgpr.has_indirect_call)
; GCN: TotalNumSgprs: 38
; GCN: NumVgprs: 50
; GCN: ScratchSize: 16
define void @indirect_use_50_vgpr() #0 {
  call void @use_50_vgpr()
  ret void
}

; GCN-LABEL: {{^}}use_80_sgpr:
; GCN:	.set use_80_sgpr.num_vgpr, 1
; GCN:	.set use_80_sgpr.num_agpr, 0
; GCN:	.set use_80_sgpr.numbered_sgpr, 80
; GCN:	.set use_80_sgpr.private_seg_size, 8
; GCN:	.set use_80_sgpr.uses_vcc, 0
; GCN:	.set use_80_sgpr.uses_flat_scratch, 0
; GCN:	.set use_80_sgpr.has_dyn_sized_stack, 0
; GCN:	.set use_80_sgpr.has_recursion, 0
; GCN:	.set use_80_sgpr.has_indirect_call, 0
; GCN: TotalNumSgprs: 84
; GCN: NumVgprs: 1
; GCN: ScratchSize: 8
define void @use_80_sgpr() #1 {
  call void asm sideeffect "", "~{s79}"() #0
  ret void
}

; GCN-LABEL: {{^}}indirect_use_80_sgpr:
; GCN:	.set indirect_use_80_sgpr.num_vgpr, max(41, use_80_sgpr.num_vgpr)
; GCN:	.set indirect_use_80_sgpr.num_agpr, max(0, use_80_sgpr.num_agpr)
; GCN:	.set indirect_use_80_sgpr.numbered_sgpr, max(34, use_80_sgpr.numbered_sgpr)
; GCN:	.set indirect_use_80_sgpr.private_seg_size, 16+(max(use_80_sgpr.private_seg_size))
; GCN:	.set indirect_use_80_sgpr.uses_vcc, or(1, use_80_sgpr.uses_vcc)
; GCN:	.set indirect_use_80_sgpr.uses_flat_scratch, or(0, use_80_sgpr.uses_flat_scratch)
; GCN:	.set indirect_use_80_sgpr.has_dyn_sized_stack, or(0, use_80_sgpr.has_dyn_sized_stack)
; GCN:	.set indirect_use_80_sgpr.has_recursion, or(0, use_80_sgpr.has_recursion)
; GCN:	.set indirect_use_80_sgpr.has_indirect_call, or(0, use_80_sgpr.has_indirect_call)
; GCN: TotalNumSgprs: 84
; GCN: NumVgprs: 41
; GCN: ScratchSize: 24
define void @indirect_use_80_sgpr() #1 {
  call void @use_80_sgpr()
  ret void
}

; GCN-LABEL: {{^}}indirect_2_level_use_80_sgpr:
; GCN:	.set indirect_2_level_use_80_sgpr.num_vgpr, max(32, indirect_use_80_sgpr.num_vgpr)
; GCN:	.set indirect_2_level_use_80_sgpr.num_agpr, max(0, indirect_use_80_sgpr.num_agpr)
; GCN:	.set indirect_2_level_use_80_sgpr.numbered_sgpr, max(33, indirect_use_80_sgpr.numbered_sgpr)
; GCN:	.set indirect_2_level_use_80_sgpr.private_seg_size, 0+(max(indirect_use_80_sgpr.private_seg_size))
; GCN:	.set indirect_2_level_use_80_sgpr.uses_vcc, or(1, indirect_use_80_sgpr.uses_vcc)
; GCN:	.set indirect_2_level_use_80_sgpr.uses_flat_scratch, or(1, indirect_use_80_sgpr.uses_flat_scratch)
; GCN:	.set indirect_2_level_use_80_sgpr.has_dyn_sized_stack, or(0, indirect_use_80_sgpr.has_dyn_sized_stack)
; GCN:	.set indirect_2_level_use_80_sgpr.has_recursion, or(0, indirect_use_80_sgpr.has_recursion)
; GCN:	.set indirect_2_level_use_80_sgpr.has_indirect_call, or(0, indirect_use_80_sgpr.has_indirect_call)
; GCN: TotalNumSgprs: 86
; GCN: NumVgprs: 41
; GCN: ScratchSize: 24
define amdgpu_kernel void @indirect_2_level_use_80_sgpr() #0 {
  call void @indirect_use_80_sgpr()
  ret void
}

; GCN-LABEL: {{^}}use_stack0:
; GCN:	.set use_stack0.num_vgpr, 1
; GCN:	.set use_stack0.num_agpr, 0
; GCN:	.set use_stack0.numbered_sgpr, 33
; GCN:	.set use_stack0.private_seg_size, 2052
; GCN:	.set use_stack0.uses_vcc, 0
; GCN:	.set use_stack0.uses_flat_scratch, 0
; GCN:	.set use_stack0.has_dyn_sized_stack, 0
; GCN:	.set use_stack0.has_recursion, 0
; GCN:	.set use_stack0.has_indirect_call, 0
; GCN: TotalNumSgprs: 37
; GCN: NumVgprs: 1
; GCN: ScratchSize: 2052
define void @use_stack0() #1 {
  %alloca = alloca [512 x i32], align 4, addrspace(5)
  call void asm sideeffect "; use $0", "v"(ptr addrspace(5) %alloca) #0
  ret void
}

; GCN-LABEL: {{^}}use_stack1:
; GCN:	.set use_stack1.num_vgpr, 1
; GCN:	.set use_stack1.num_agpr, 0
; GCN:	.set use_stack1.numbered_sgpr, 33
; GCN:	.set use_stack1.private_seg_size, 404
; GCN:	.set use_stack1.uses_vcc, 0
; GCN:	.set use_stack1.uses_flat_scratch, 0
; GCN:	.set use_stack1.has_dyn_sized_stack, 0
; GCN:	.set use_stack1.has_recursion, 0
; GCN:	.set use_stack1.has_indirect_call, 0
; GCN: TotalNumSgprs: 37
; GCN: NumVgprs: 1
; GCN: ScratchSize: 404
define void @use_stack1() #1 {
  %alloca = alloca [100 x i32], align 4, addrspace(5)
  call void asm sideeffect "; use $0", "v"(ptr addrspace(5) %alloca) #0
  ret void
}

; GCN-LABEL: {{^}}indirect_use_stack:
; GCN:	.set indirect_use_stack.num_vgpr, max(41, use_stack0.num_vgpr)
; GCN:	.set indirect_use_stack.num_agpr, max(0, use_stack0.num_agpr)
; GCN:	.set indirect_use_stack.numbered_sgpr, max(34, use_stack0.numbered_sgpr)
; GCN:	.set indirect_use_stack.private_seg_size, 80+(max(use_stack0.private_seg_size))
; GCN:	.set indirect_use_stack.uses_vcc, or(1, use_stack0.uses_vcc)
; GCN:	.set indirect_use_stack.uses_flat_scratch, or(0, use_stack0.uses_flat_scratch)
; GCN:	.set indirect_use_stack.has_dyn_sized_stack, or(0, use_stack0.has_dyn_sized_stack)
; GCN:	.set indirect_use_stack.has_recursion, or(0, use_stack0.has_recursion)
; GCN:	.set indirect_use_stack.has_indirect_call, or(0, use_stack0.has_indirect_call)
; GCN: TotalNumSgprs: 38
; GCN: NumVgprs: 41
; GCN: ScratchSize: 2132
define void @indirect_use_stack() #1 {
  %alloca = alloca [16 x i32], align 4, addrspace(5)
  call void asm sideeffect "; use $0", "v"(ptr addrspace(5) %alloca) #0
  call void @use_stack0()
  ret void
}

; GCN-LABEL: {{^}}indirect_2_level_use_stack:
; GCN:	.set indirect_2_level_use_stack.num_vgpr, max(32, indirect_use_stack.num_vgpr)
; GCN:	.set indirect_2_level_use_stack.num_agpr, max(0, indirect_use_stack.num_agpr)
; GCN:	.set indirect_2_level_use_stack.numbered_sgpr, max(33, indirect_use_stack.numbered_sgpr)
; GCN:	.set indirect_2_level_use_stack.private_seg_size, 0+(max(indirect_use_stack.private_seg_size))
; GCN:	.set indirect_2_level_use_stack.uses_vcc, or(1, indirect_use_stack.uses_vcc)
; GCN:	.set indirect_2_level_use_stack.uses_flat_scratch, or(1, indirect_use_stack.uses_flat_scratch)
; GCN:	.set indirect_2_level_use_stack.has_dyn_sized_stack, or(0, indirect_use_stack.has_dyn_sized_stack)
; GCN:	.set indirect_2_level_use_stack.has_recursion, or(0, indirect_use_stack.has_recursion)
; GCN:	.set indirect_2_level_use_stack.has_indirect_call, or(0, indirect_use_stack.has_indirect_call)
; GCN: TotalNumSgprs: 40
; GCN: NumVgprs: 41
; GCN: ScratchSize: 2132
define amdgpu_kernel void @indirect_2_level_use_stack() #0 {
  call void @indirect_use_stack()
  ret void
}


; Should be maximum of callee usage
; GCN-LABEL: {{^}}multi_call_use_use_stack:
; GCN:	.set multi_call_use_use_stack.num_vgpr, max(41, use_stack0.num_vgpr, use_stack1.num_vgpr)
; GCN:	.set multi_call_use_use_stack.num_agpr, max(0, use_stack0.num_agpr, use_stack1.num_agpr)
; GCN:	.set multi_call_use_use_stack.numbered_sgpr, max(42, use_stack0.numbered_sgpr, use_stack1.numbered_sgpr)
; GCN:	.set multi_call_use_use_stack.private_seg_size, 0+(max(use_stack0.private_seg_size, use_stack1.private_seg_size))
; GCN:	.set multi_call_use_use_stack.uses_vcc, or(1, use_stack0.uses_vcc, use_stack1.uses_vcc)
; GCN:	.set multi_call_use_use_stack.uses_flat_scratch, or(1, use_stack0.uses_flat_scratch, use_stack1.uses_flat_scratch)
; GCN:	.set multi_call_use_use_stack.has_dyn_sized_stack, or(0, use_stack0.has_dyn_sized_stack, use_stack1.has_dyn_sized_stack)
; GCN:	.set multi_call_use_use_stack.has_recursion, or(0, use_stack0.has_recursion, use_stack1.has_recursion)
; GCN:	.set multi_call_use_use_stack.has_indirect_call, or(0, use_stack0.has_indirect_call, use_stack1.has_indirect_call)
; GCN: TotalNumSgprs: 48
; GCN: NumVgprs: 41
; GCN: ScratchSize: 2052
define amdgpu_kernel void @multi_call_use_use_stack() #0 {
  call void @use_stack0()
  call void @use_stack1()
  ret void
}

declare void @external() #0

; GCN-LABEL: {{^}}multi_call_with_external:
; GCN:	.set multi_call_with_external.num_vgpr, max(41, amdgpu.max_num_vgpr)
; GCN:	.set multi_call_with_external.num_agpr, max(0, amdgpu.max_num_agpr)
; GCN:	.set multi_call_with_external.numbered_sgpr, max(42, amdgpu.max_num_sgpr)
; GCN:	.set multi_call_with_external.private_seg_size, 0+(max(use_stack0.private_seg_size, use_stack1.private_seg_size))
; GCN:	.set multi_call_with_external.uses_vcc, 1
; GCN:	.set multi_call_with_external.uses_flat_scratch, 1
; GCN:	.set multi_call_with_external.has_dyn_sized_stack, 1
; GCN:	.set multi_call_with_external.has_recursion, 0
; GCN:	.set multi_call_with_external.has_indirect_call, 1
; GCN: TotalNumSgprs: multi_call_with_external.numbered_sgpr+6
; GCN: NumVgprs: multi_call_with_external.num_vgpr
; GCN: ScratchSize: 2052
define amdgpu_kernel void @multi_call_with_external() #0 {
  call void @use_stack0()
  call void @use_stack1()
  call void @external()
  ret void
}

; GCN-LABEL: {{^}}multi_call_with_external_and_duplicates:
; GCN:	.set multi_call_with_external_and_duplicates.num_vgpr, max(41, amdgpu.max_num_vgpr)
; GCN:	.set multi_call_with_external_and_duplicates.num_agpr, max(0, amdgpu.max_num_agpr)
; GCN:	.set multi_call_with_external_and_duplicates.numbered_sgpr, max(44, amdgpu.max_num_sgpr)
; GCN:	.set multi_call_with_external_and_duplicates.private_seg_size, 0+(max(use_stack0.private_seg_size, use_stack1.private_seg_size))
; GCN:	.set multi_call_with_external_and_duplicates.uses_vcc, 1
; GCN:	.set multi_call_with_external_and_duplicates.uses_flat_scratch, 1
; GCN:	.set multi_call_with_external_and_duplicates.has_dyn_sized_stack, 1
; GCN:	.set multi_call_with_external_and_duplicates.has_recursion, 0
; GCN:	.set multi_call_with_external_and_duplicates.has_indirect_call, 1
; GCN: TotalNumSgprs: multi_call_with_external_and_duplicates.numbered_sgpr+6
; GCN: NumVgprs: multi_call_with_external_and_duplicates.num_vgpr
; GCN: ScratchSize: 2052
define amdgpu_kernel void @multi_call_with_external_and_duplicates() #0 {
  call void @use_stack0()
  call void @use_stack0()
  call void @use_stack1()
  call void @use_stack1()
  call void @external()
  call void @external()
  ret void
}

; GCN-LABEL: {{^}}usage_external:
; GCN:	.set usage_external.num_vgpr, max(32, amdgpu.max_num_vgpr)
; GCN:	.set usage_external.num_agpr, max(0, amdgpu.max_num_agpr)
; GCN:	.set usage_external.numbered_sgpr, max(33, amdgpu.max_num_sgpr)
; GCN:	.set usage_external.private_seg_size, 0
; GCN:	.set usage_external.uses_vcc, 1
; GCN:	.set usage_external.uses_flat_scratch, 1
; GCN:	.set usage_external.has_dyn_sized_stack, 1
; GCN:	.set usage_external.has_recursion, 0
; GCN:	.set usage_external.has_indirect_call, 1
; GCN: TotalNumSgprs: usage_external.numbered_sgpr+6
; GCN: NumVgprs: usage_external.num_vgpr
; GCN: ScratchSize: 0
define amdgpu_kernel void @usage_external() #0 {
  call void @external()
  ret void
}

declare void @external_recurse() #2

; GCN-LABEL: {{^}}usage_external_recurse:
; GCN:	.set usage_external_recurse.num_vgpr, max(32, amdgpu.max_num_vgpr)
; GCN:	.set usage_external_recurse.num_agpr, max(0, amdgpu.max_num_agpr)
; GCN:	.set usage_external_recurse.numbered_sgpr, max(33, amdgpu.max_num_sgpr)
; GCN:	.set usage_external_recurse.private_seg_size, 0
; GCN:	.set usage_external_recurse.uses_vcc, 1
; GCN:	.set usage_external_recurse.uses_flat_scratch, 1
; GCN:	.set usage_external_recurse.has_dyn_sized_stack, 1
; GCN:	.set usage_external_recurse.has_recursion, 1
; GCN:	.set usage_external_recurse.has_indirect_call, 1
; GCN: TotalNumSgprs: usage_external_recurse.numbered_sgpr+6
; GCN: NumVgprs: usage_external_recurse.num_vgpr
; GCN: ScratchSize: 0
define amdgpu_kernel void @usage_external_recurse() #0 {
  call void @external_recurse()
  ret void
}

; GCN-LABEL: {{^}}direct_recursion_use_stack:
; GCN: .set direct_recursion_use_stack.num_vgpr, 41
; GCN: .set direct_recursion_use_stack.num_agpr, 0
; GCN: .set direct_recursion_use_stack.numbered_sgpr, 36
; GCN: .set direct_recursion_use_stack.private_seg_size, 2064
; GCN: .set direct_recursion_use_stack.uses_vcc, 1
; GCN: .set direct_recursion_use_stack.uses_flat_scratch, 0
; GCN: .set direct_recursion_use_stack.has_dyn_sized_stack, 0
; GCN: .set direct_recursion_use_stack.has_recursion, 1
; GCN: .set direct_recursion_use_stack.has_indirect_call, 0
; GCN: TotalNumSgprs: 40
; GCN: NumVgprs: 41
; GCN: ScratchSize: 2064
define void @direct_recursion_use_stack(i32 %val) #2 {
  %alloca = alloca [512 x i32], align 4, addrspace(5)
  call void asm sideeffect "; use $0", "v"(ptr addrspace(5) %alloca) #0
  %cmp = icmp eq i32 %val, 0
  br i1 %cmp, label %ret, label %call

call:
  %val.sub1 = sub i32 %val, 1
  call void @direct_recursion_use_stack(i32 %val.sub1)
  br label %ret

ret:
  ret void
}

; GCN-LABEL: {{^}}usage_direct_recursion:
; GCN:  .set usage_direct_recursion.num_vgpr, max(32, direct_recursion_use_stack.num_vgpr)
; GCN:  .set usage_direct_recursion.num_agpr, max(0, direct_recursion_use_stack.num_agpr)
; GCN:  .set usage_direct_recursion.numbered_sgpr, max(33, direct_recursion_use_stack.numbered_sgpr)
; GCN:  .set usage_direct_recursion.private_seg_size, 0+(max(direct_recursion_use_stack.private_seg_size))
; GCN:  .set usage_direct_recursion.uses_vcc, or(1, direct_recursion_use_stack.uses_vcc)
; GCN:  .set usage_direct_recursion.uses_flat_scratch, or(1, direct_recursion_use_stack.uses_flat_scratch)
; GCN:  .set usage_direct_recursion.has_dyn_sized_stack, or(0, direct_recursion_use_stack.has_dyn_sized_stack)
; GCN:  .set usage_direct_recursion.has_recursion, or(1, direct_recursion_use_stack.has_recursion)
; GCN:  .set usage_direct_recursion.has_indirect_call, or(0, direct_recursion_use_stack.has_indirect_call)
; GCN: TotalNumSgprs: 42
; GCN: NumVgprs: 41
; GCN: ScratchSize: 2064
define amdgpu_kernel void @usage_direct_recursion(i32 %n) #0 {
  call void @direct_recursion_use_stack(i32 %n)
  ret void
}

; Make sure there's no assert when a sgpr96 is used.
; GCN-LABEL: {{^}}count_use_sgpr96_external_call
; GCN:	.set count_use_sgpr96_external_call.num_vgpr, max(32, amdgpu.max_num_vgpr)
; GCN:	.set count_use_sgpr96_external_call.num_agpr, max(0, amdgpu.max_num_agpr)
; GCN:	.set count_use_sgpr96_external_call.numbered_sgpr, max(33, amdgpu.max_num_sgpr)
; GCN:	.set count_use_sgpr96_external_call.private_seg_size, 0
; GCN:	.set count_use_sgpr96_external_call.uses_vcc, 1
; GCN:	.set count_use_sgpr96_external_call.uses_flat_scratch, 1
; GCN:	.set count_use_sgpr96_external_call.has_dyn_sized_stack, 1
; GCN:	.set count_use_sgpr96_external_call.has_recursion, 0
; GCN:	.set count_use_sgpr96_external_call.has_indirect_call, 1
; GCN: TotalNumSgprs: count_use_sgpr96_external_call.numbered_sgpr+6
; GCN: NumVgprs: count_use_sgpr96_external_call.num_vgpr
; GCN: ScratchSize: 0
define amdgpu_kernel void @count_use_sgpr96_external_call()  {
entry:
  tail call void asm sideeffect "; sgpr96 $0", "s"(<3 x i32> <i32 10, i32 11, i32 12>) #1
  call void @external()
  ret void
}

; Make sure there's no assert when a sgpr160 is used.
; GCN-LABEL: {{^}}count_use_sgpr160_external_call
; GCN:	.set count_use_sgpr160_external_call.num_vgpr, max(32, amdgpu.max_num_vgpr)
; GCN:	.set count_use_sgpr160_external_call.num_agpr, max(0, amdgpu.max_num_agpr)
; GCN:	.set count_use_sgpr160_external_call.numbered_sgpr, max(33, amdgpu.max_num_sgpr)
; GCN:	.set count_use_sgpr160_external_call.private_seg_size, 0
; GCN:	.set count_use_sgpr160_external_call.uses_vcc, 1
; GCN:	.set count_use_sgpr160_external_call.uses_flat_scratch, 1
; GCN:	.set count_use_sgpr160_external_call.has_dyn_sized_stack, 1
; GCN:	.set count_use_sgpr160_external_call.has_recursion, 0
; GCN:	.set count_use_sgpr160_external_call.has_indirect_call, 1
; GCN: TotalNumSgprs: count_use_sgpr160_external_call.numbered_sgpr+6
; GCN: NumVgprs: count_use_sgpr160_external_call.num_vgpr
; GCN: ScratchSize: 0
define amdgpu_kernel void @count_use_sgpr160_external_call()  {
entry:
  tail call void asm sideeffect "; sgpr160 $0", "s"(<5 x i32> <i32 10, i32 11, i32 12, i32 13, i32 14>) #1
  call void @external()
  ret void
}

; Make sure there's no assert when a vgpr160 is used.
; GCN-LABEL: {{^}}count_use_vgpr160_external_call
; GCN:	.set count_use_vgpr160_external_call.num_vgpr, max(32, amdgpu.max_num_vgpr)
; GCN:	.set count_use_vgpr160_external_call.num_agpr, max(0, amdgpu.max_num_agpr)
; GCN:	.set count_use_vgpr160_external_call.numbered_sgpr, max(33, amdgpu.max_num_sgpr)
; GCN:	.set count_use_vgpr160_external_call.private_seg_size, 0
; GCN:	.set count_use_vgpr160_external_call.uses_vcc, 1
; GCN:	.set count_use_vgpr160_external_call.uses_flat_scratch, 1
; GCN:	.set count_use_vgpr160_external_call.has_dyn_sized_stack, 1
; GCN:	.set count_use_vgpr160_external_call.has_recursion, 0
; GCN:	.set count_use_vgpr160_external_call.has_indirect_call, 1
; GCN: TotalNumSgprs: count_use_vgpr160_external_call.numbered_sgpr+6
; GCN: NumVgprs: count_use_vgpr160_external_call.num_vgpr
; GCN: ScratchSize: 0
define amdgpu_kernel void @count_use_vgpr160_external_call()  {
entry:
  tail call void asm sideeffect "; vgpr160 $0", "v"(<5 x i32> <i32 10, i32 11, i32 12, i32 13, i32 14>) #1
  call void @external()
  ret void
}

; Added at the of the .s are the module level maximums
; GCN:	.set amdgpu.max_num_vgpr, 50
; GCN:	.set amdgpu.max_num_agpr, 0
; GCN:	.set amdgpu.max_num_sgpr, 80

attributes #0 = { nounwind noinline norecurse }
attributes #1 = { nounwind noinline norecurse }
attributes #2 = { nounwind noinline }
