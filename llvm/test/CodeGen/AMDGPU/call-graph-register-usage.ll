; RUN: sed 's/CODE_OBJECT_VERSION/400/g' %s | llc -mtriple=amdgcn-amd-amdhsa -enable-ipra=0 -verify-machineinstrs | FileCheck -check-prefixes=GCN,CI %s
; RUN: sed 's/CODE_OBJECT_VERSION/500/g' %s | llc -mtriple=amdgcn-amd-amdhsa -enable-ipra=0 -verify-machineinstrs | FileCheck -check-prefixes=GCN-V5 %s
; RUN: sed 's/CODE_OBJECT_VERSION/600/g' %s | llc -mtriple=amdgcn-amd-amdhsa -enable-ipra=0 -verify-machineinstrs | FileCheck -check-prefixes=GCN-V5 %s
; RUN: sed 's/CODE_OBJECT_VERSION/400/g' %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=fiji -enable-ipra=0 -verify-machineinstrs | FileCheck -check-prefixes=GCN,VI,VI-NOBUG %s
; RUN: sed 's/CODE_OBJECT_VERSION/400/g' %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=iceland -enable-ipra=0 -verify-machineinstrs | FileCheck -check-prefixes=GCN,VI,VI-BUG %s

; Make sure to run a GPU with the SGPR allocation bug.

; GCN-LABEL: {{^}}use_vcc:
; GCN: ; TotalNumSgprs: 34
; GCN: ; NumVgprs: 0
define void @use_vcc() #1 {
  call void asm sideeffect "", "~{vcc}" () #0
  ret void
}

; GCN-LABEL: {{^}}indirect_use_vcc:
; GCN: s_mov_b32 s4, s33
; GCN: v_writelane_b32 v40, s4, 2
; GCN: v_writelane_b32 v40, s30, 0
; GCN: v_writelane_b32 v40, s31, 1
; GCN: s_swappc_b64
; GCN: v_readlane_b32 s31, v40, 1
; GCN: v_readlane_b32 s30, v40, 0
; GCN: v_readlane_b32 s4, v40, 2
; GCN: s_mov_b32 s33, s4
; GCN: s_setpc_b64 s[30:31]
; GCN: ; TotalNumSgprs: 36
; GCN: ; NumVgprs: 41
define void @indirect_use_vcc() #1 {
  call void @use_vcc()
  ret void
}

; GCN-LABEL: {{^}}indirect_2level_use_vcc_kernel:
; CI: ; TotalNumSgprs: 38
; VI-NOBUG: ; TotalNumSgprs: 40
; VI-BUG: ; TotalNumSgprs: 96
; GCN: ; NumVgprs: 41
define amdgpu_kernel void @indirect_2level_use_vcc_kernel(ptr addrspace(1) %out) #0 {
  call void @indirect_use_vcc()
  ret void
}

; GCN-LABEL: {{^}}use_flat_scratch:
; CI: ; TotalNumSgprs: 36
; VI: ; TotalNumSgprs: 38
; GCN: ; NumVgprs: 0
define void @use_flat_scratch() #1 {
  call void asm sideeffect "", "~{flat_scratch}" () #0
  ret void
}

; GCN-LABEL: {{^}}indirect_use_flat_scratch:
; CI: ; TotalNumSgprs: 38
; VI: ; TotalNumSgprs: 40
; GCN: ; NumVgprs: 41
define void @indirect_use_flat_scratch() #1 {
  call void @use_flat_scratch()
  ret void
}

; GCN-LABEL: {{^}}indirect_2level_use_flat_scratch_kernel:
; CI: ; TotalNumSgprs: 38
; VI-NOBUG: ; TotalNumSgprs: 40
; VI-BUG: ; TotalNumSgprs: 96
; GCN: ; NumVgprs: 41
define amdgpu_kernel void @indirect_2level_use_flat_scratch_kernel(ptr addrspace(1) %out) #0 {
  call void @indirect_use_flat_scratch()
  ret void
}

; GCN-LABEL: {{^}}use_10_vgpr:
; GCN: ; NumVgprs: 10
define void @use_10_vgpr() #1 {
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4}"() #0
  call void asm sideeffect "", "~{v5},~{v6},~{v7},~{v8},~{v9}"() #0
  ret void
}

; GCN-LABEL: {{^}}indirect_use_10_vgpr:
; GCN: ; NumVgprs: 41
define void @indirect_use_10_vgpr() #0 {
  call void @use_10_vgpr()
  ret void
}

; GCN-LABEL: {{^}}indirect_2_level_use_10_vgpr:
; GCN: ; NumVgprs: 41
define amdgpu_kernel void @indirect_2_level_use_10_vgpr() #0 {
  call void @indirect_use_10_vgpr()
  ret void
}

; GCN-LABEL: {{^}}use_50_vgpr:
; GCN: ; NumVgprs: 50
define void @use_50_vgpr() #1 {
  call void asm sideeffect "", "~{v49}"() #0
  ret void
}

; GCN-LABEL: {{^}}indirect_use_50_vgpr:
; GCN: ; NumVgprs: 50
define void @indirect_use_50_vgpr() #0 {
  call void @use_50_vgpr()
  ret void
}

; GCN-LABEL: {{^}}use_80_sgpr:
; GCN: ; TotalNumSgprs: 80
define void @use_80_sgpr() #1 {
  call void asm sideeffect "", "~{s79}"() #0
  ret void
}

; GCN-LABEL: {{^}}indirect_use_80_sgpr:
; GCN: ; TotalNumSgprs: 82
define void @indirect_use_80_sgpr() #1 {
  call void @use_80_sgpr()
  ret void
}

; GCN-LABEL: {{^}}indirect_2_level_use_80_sgpr:
; CI: ; TotalNumSgprs: 84
; VI-NOBUG: ; TotalNumSgprs: 86
; VI-BUG: ; TotalNumSgprs: 96
define amdgpu_kernel void @indirect_2_level_use_80_sgpr() #0 {
  call void @indirect_use_80_sgpr()
  ret void
}


; GCN-LABEL: {{^}}use_stack0:
; GCN: ScratchSize: 2052
define void @use_stack0() #1 {
  %alloca = alloca [512 x i32], align 4, addrspace(5)
  call void asm sideeffect "; use $0", "v"(ptr addrspace(5) %alloca) #0
  ret void
}

; GCN-LABEL: {{^}}use_stack1:
; GCN: ScratchSize: 404
define void @use_stack1() #1 {
  %alloca = alloca [100 x i32], align 4, addrspace(5)
  call void asm sideeffect "; use $0", "v"(ptr addrspace(5) %alloca) #0
  ret void
}

; GCN-LABEL: {{^}}indirect_use_stack:
; GCN: ScratchSize: 2132
define void @indirect_use_stack() #1 {
  %alloca = alloca [16 x i32], align 4, addrspace(5)
  call void asm sideeffect "; use $0", "v"(ptr addrspace(5) %alloca) #0
  call void @use_stack0()
  ret void
}

; GCN-LABEL: {{^}}indirect_2_level_use_stack:
; GCN: ScratchSize: 2132
define amdgpu_kernel void @indirect_2_level_use_stack() #0 {
  call void @indirect_use_stack()
  ret void
}


; Should be maximum of callee usage
; GCN-LABEL: {{^}}multi_call_use_use_stack:
; GCN: ScratchSize: 2052
define amdgpu_kernel void @multi_call_use_use_stack() #0 {
  call void @use_stack0()
  call void @use_stack1()
  ret void
}


declare void @external() #0

; GCN-LABEL: {{^}}usage_external:
; TotalNumSgprs: 48
; NumVgprs: 24
; GCN: ScratchSize: 16384
;
; GCN-V5-LABEL: {{^}}usage_external:
; GCN-V5: ScratchSize: 0
define amdgpu_kernel void @usage_external() #0 {
  call void @external()
  ret void
}

declare void @external_recurse() #2

; GCN-LABEL: {{^}}usage_external_recurse:
; TotalNumSgprs: 48
; NumVgprs: 24
; GCN: ScratchSize: 16384
;
; GCN-V5-LABEL: {{^}}usage_external_recurse:
; GCN-V5: ScratchSize: 0
define amdgpu_kernel void @usage_external_recurse() #0 {
  call void @external_recurse()
  ret void
}

; GCN-LABEL: {{^}}direct_recursion_use_stack:
; GCN: ScratchSize: 18448{{$}}
;
; GCN-V5-LABEL: {{^}}direct_recursion_use_stack:
; GCN-V5: ScratchSize: 2064{{$}}
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
; GCN: .amdhsa_private_segment_fixed_size 18448
;
; GCN-V5-LABEL: {{^}}usage_direct_recursion:
; GCN-V5: .amdhsa_private_segment_fixed_size 2064{{$}}
define amdgpu_kernel void @usage_direct_recursion(i32 %n) #0 {
  call void @direct_recursion_use_stack(i32 %n)
  ret void
}

; Make sure there's no assert when a sgpr96 is used.
; GCN-LABEL: {{^}}count_use_sgpr96_external_call
; GCN: ; sgpr96 s[{{[0-9]+}}:{{[0-9]+}}]
; GCN: .set count_use_sgpr96_external_call.num_vgpr, max(0, amdgpu.max_num_vgpr)
; GCN: .set count_use_sgpr96_external_call.numbered_sgpr, max(33, amdgpu.max_num_sgpr)
; CI: TotalNumSgprs: count_use_sgpr96_external_call.numbered_sgpr+4
; VI-BUG: TotalNumSgprs: 96
; GCN: NumVgprs: count_use_sgpr96_external_call.num_vgpr
define amdgpu_kernel void @count_use_sgpr96_external_call()  {
entry:
  tail call void asm sideeffect "; sgpr96 $0", "s"(<3 x i32> <i32 10, i32 11, i32 12>) #1
  call void @external()
  ret void
}

; Make sure there's no assert when a sgpr160 is used.
; GCN-LABEL: {{^}}count_use_sgpr160_external_call
; GCN: ; sgpr160 s[{{[0-9]+}}:{{[0-9]+}}]
; GCN: .set count_use_sgpr160_external_call.num_vgpr, max(0, amdgpu.max_num_vgpr)
; GCN: .set count_use_sgpr160_external_call.numbered_sgpr, max(33, amdgpu.max_num_sgpr)
; CI: TotalNumSgprs: count_use_sgpr160_external_call.numbered_sgpr+4
; VI-BUG: TotalNumSgprs: 96
; GCN: NumVgprs: count_use_sgpr160_external_call.num_vgpr
define amdgpu_kernel void @count_use_sgpr160_external_call()  {
entry:
  tail call void asm sideeffect "; sgpr160 $0", "s"(<5 x i32> <i32 10, i32 11, i32 12, i32 13, i32 14>) #1
  call void @external()
  ret void
}

; Make sure there's no assert when a vgpr160 is used.
; GCN-LABEL: {{^}}count_use_vgpr160_external_call
; GCN: ; vgpr160 v[{{[0-9]+}}:{{[0-9]+}}]
; GCN: .set count_use_vgpr160_external_call.num_vgpr, max(5, amdgpu.max_num_vgpr)
; GCN: .set count_use_vgpr160_external_call.numbered_sgpr, max(33, amdgpu.max_num_sgpr)
; CI: TotalNumSgprs: count_use_vgpr160_external_call.numbered_sgpr+4
; VI-BUG: TotalNumSgprs: 96
; GCN: NumVgprs: count_use_vgpr160_external_call.num_vgpr
define amdgpu_kernel void @count_use_vgpr160_external_call()  {
entry:
  tail call void asm sideeffect "; vgpr160 $0", "v"(<5 x i32> <i32 10, i32 11, i32 12, i32 13, i32 14>) #1
  call void @external()
  ret void
}

; GCN: .set amdgpu.max_num_vgpr, 50
; GCN: .set amdgpu.max_num_agpr, 0
; GCN: .set amdgpu.max_num_sgpr, 80

; GCN-LABEL: amdhsa.kernels:
; GCN:      .name: count_use_sgpr96_external_call
; CI:       .sgpr_count: 84
; VI-NOBUG: .sgpr_count: 86
; VI-BUG:   .sgpr_count: 96
; GCN:      .vgpr_count: 50
; GCN:      .name: count_use_sgpr160_external_call
; CI:       .sgpr_count: 84
; VI-NOBUG: .sgpr_count: 86
; VI-BUG:   .sgpr_count: 96
; GCN:      .vgpr_count: 50
; GCN:      .name: count_use_vgpr160_external_call
; CI:       .sgpr_count: 84
; VI-NOBUG: .sgpr_count: 86
; VI-BUG:   .sgpr_count: 96
; GCN:      .vgpr_count: 50

attributes #0 = { nounwind noinline norecurse "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" }
attributes #1 = { nounwind noinline norecurse "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" }
attributes #2 = { nounwind noinline }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 CODE_OBJECT_VERSION}
