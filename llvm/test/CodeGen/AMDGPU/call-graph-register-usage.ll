; RUN: sed 's/CODE_OBJECT_VERSION/400/g' %s | llc -mtriple=amdgcn-amd-amdhsa -enable-ipra=0 -verify-machineinstrs | FileCheck -check-prefixes=GCN,CI %s
; RUN: sed 's/CODE_OBJECT_VERSION/500/g' %s | llc -mtriple=amdgcn-amd-amdhsa -enable-ipra=0 -verify-machineinstrs | FileCheck -check-prefixes=GCN-V5 %s
; RUN: sed 's/CODE_OBJECT_VERSION/600/g' %s | llc -mtriple=amdgcn-amd-amdhsa -enable-ipra=0 -verify-machineinstrs | FileCheck -check-prefixes=GCN-V5 %s
; RUN: sed 's/CODE_OBJECT_VERSION/400/g' %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=fiji -enable-ipra=0 -verify-machineinstrs | FileCheck -check-prefixes=GCN,VI,VI-NOBUG %s
; RUN: sed 's/CODE_OBJECT_VERSION/400/g' %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=iceland -enable-ipra=0 -verify-machineinstrs | FileCheck -check-prefixes=GCN,VI,VI-BUG %s

; Make sure to run a GPU with the SGPR allocation bug.

; GCN-LABEL: {{^}}use_vcc:
; GCN: ; NumSgprs: 34
; GCN: ; NumVgprs: 0
define void @use_vcc() nounwind noinline norecurse {
  call void asm sideeffect "", "~{vcc}" () nounwind noinline norecurse
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
; GCN: ; NumSgprs: 36
; GCN: ; NumVgprs: 41
define void @indirect_use_vcc() nounwind noinline norecurse {
  call void @use_vcc()
  ret void
}

; GCN-LABEL: {{^}}indirect_2level_use_vcc_kernel:
; CI: ; NumSgprs: 38
; VI-NOBUG: ; NumSgprs: 40
; VI-BUG: ; NumSgprs: 96
; GCN: ; NumVgprs: 41
define amdgpu_kernel void @indirect_2level_use_vcc_kernel(ptr addrspace(1) %out) nounwind noinline norecurse {
  call void @indirect_use_vcc()
  ret void
}

; GCN-LABEL: {{^}}use_flat_scratch:
; CI: ; NumSgprs: 36
; VI: ; NumSgprs: 38
; GCN: ; NumVgprs: 0
define void @use_flat_scratch() nounwind noinline norecurse {
  call void asm sideeffect "", "~{flat_scratch}" () nounwind noinline norecurse
  ret void
}

; GCN-LABEL: {{^}}indirect_use_flat_scratch:
; CI: ; NumSgprs: 38
; VI: ; NumSgprs: 40
; GCN: ; NumVgprs: 41
define void @indirect_use_flat_scratch() nounwind noinline norecurse {
  call void @use_flat_scratch()
  ret void
}

; GCN-LABEL: {{^}}indirect_2level_use_flat_scratch_kernel:
; CI: ; NumSgprs: 38
; VI-NOBUG: ; NumSgprs: 40
; VI-BUG: ; NumSgprs: 96
; GCN: ; NumVgprs: 41
define amdgpu_kernel void @indirect_2level_use_flat_scratch_kernel(ptr addrspace(1) %out) nounwind noinline norecurse {
  call void @indirect_use_flat_scratch()
  ret void
}

; GCN-LABEL: {{^}}use_10_vgpr:
; GCN: ; NumVgprs: 10
define void @use_10_vgpr() nounwind noinline norecurse {
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4}"() nounwind noinline norecurse
  call void asm sideeffect "", "~{v5},~{v6},~{v7},~{v8},~{v9}"() nounwind noinline norecurse
  ret void
}

; GCN-LABEL: {{^}}indirect_use_10_vgpr:
; GCN: ; NumVgprs: 41
define void @indirect_use_10_vgpr() nounwind noinline norecurse {
  call void @use_10_vgpr()
  ret void
}

; GCN-LABEL: {{^}}indirect_2_level_use_10_vgpr:
; GCN: ; NumVgprs: 41
define amdgpu_kernel void @indirect_2_level_use_10_vgpr() nounwind noinline norecurse {
  call void @indirect_use_10_vgpr()
  ret void
}

; GCN-LABEL: {{^}}use_50_vgpr:
; GCN: ; NumVgprs: 50
define void @use_50_vgpr() nounwind noinline norecurse {
  call void asm sideeffect "", "~{v49}"() nounwind noinline norecurse
  ret void
}

; GCN-LABEL: {{^}}indirect_use_50_vgpr:
; GCN: ; NumVgprs: 50
define void @indirect_use_50_vgpr() nounwind noinline norecurse {
  call void @use_50_vgpr()
  ret void
}

; GCN-LABEL: {{^}}use_80_sgpr:
; GCN: ; NumSgprs: 80
define void @use_80_sgpr() nounwind noinline norecurse {
  call void asm sideeffect "", "~{s79}"() nounwind noinline norecurse
  ret void
}

; GCN-LABEL: {{^}}indirect_use_80_sgpr:
; GCN: ; NumSgprs: 82
define void @indirect_use_80_sgpr() nounwind noinline norecurse {
  call void @use_80_sgpr()
  ret void
}

; GCN-LABEL: {{^}}indirect_2_level_use_80_sgpr:
; CI: ; NumSgprs: 84
; VI-NOBUG: ; NumSgprs: 86
; VI-BUG: ; NumSgprs: 96
define amdgpu_kernel void @indirect_2_level_use_80_sgpr() nounwind noinline norecurse {
  call void @indirect_use_80_sgpr()
  ret void
}


; GCN-LABEL: {{^}}use_stack0:
; GCN: ScratchSize: 2052
define void @use_stack0() nounwind noinline norecurse {
  %alloca = alloca [512 x i32], align 4, addrspace(5)
  call void asm sideeffect "; use $0", "v"(ptr addrspace(5) %alloca) nounwind noinline norecurse
  ret void
}

; GCN-LABEL: {{^}}use_stack1:
; GCN: ScratchSize: 404
define void @use_stack1() nounwind noinline norecurse {
  %alloca = alloca [100 x i32], align 4, addrspace(5)
  call void asm sideeffect "; use $0", "v"(ptr addrspace(5) %alloca) nounwind noinline norecurse
  ret void
}

; GCN-LABEL: {{^}}indirect_use_stack:
; GCN: ScratchSize: 2132
define void @indirect_use_stack() nounwind noinline norecurse {
  %alloca = alloca [16 x i32], align 4, addrspace(5)
  call void asm sideeffect "; use $0", "v"(ptr addrspace(5) %alloca) nounwind noinline norecurse
  call void @use_stack0()
  ret void
}

; GCN-LABEL: {{^}}indirect_2_level_use_stack:
; GCN: ScratchSize: 2132
define amdgpu_kernel void @indirect_2_level_use_stack() nounwind noinline norecurse {
  call void @indirect_use_stack()
  ret void
}


; Should be maximum of callee usage
; GCN-LABEL: {{^}}multi_call_use_use_stack:
; GCN: ScratchSize: 2052
define amdgpu_kernel void @multi_call_use_use_stack() nounwind noinline norecurse {
  call void @use_stack0()
  call void @use_stack1()
  ret void
}


declare void @external() nounwind noinline norecurse

; GCN-LABEL: {{^}}usage_external:
; NumSgprs: 48
; NumVgprs: 24
; GCN: ScratchSize: 16384
;
; GCN-V5-LABEL: {{^}}usage_external:
; GCN-V5: ScratchSize: 0
define amdgpu_kernel void @usage_external() nounwind noinline norecurse {
  call void @external()
  ret void
}

declare void @external_recurse() nounwind noinline

; GCN-LABEL: {{^}}usage_external_recurse:
; NumSgprs: 48
; NumVgprs: 24
; GCN: ScratchSize: 16384
;
; GCN-V5-LABEL: {{^}}usage_external_recurse:
; GCN-V5: ScratchSize: 0
define amdgpu_kernel void @usage_external_recurse() nounwind noinline norecurse {
  call void @external_recurse()
  ret void
}

; GCN-LABEL: {{^}}direct_recursion_use_stack:
; GCN: ScratchSize: 18448{{$}}
;
; GCN-V5-LABEL: {{^}}direct_recursion_use_stack:
; GCN-V5: ScratchSize: 2064{{$}}
define void @direct_recursion_use_stack(i32 %val) nounwind noinline {
  %alloca = alloca [512 x i32], align 4, addrspace(5)
  call void asm sideeffect "; use $0", "v"(ptr addrspace(5) %alloca) nounwind noinline norecurse
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
define amdgpu_kernel void @usage_direct_recursion(i32 %n) nounwind noinline norecurse {
  call void @direct_recursion_use_stack(i32 %n)
  ret void
}

; Make sure there's no assert when a sgpr96 is used.
; GCN-LABEL: {{^}}count_use_sgpr96_external_call
; GCN: ; sgpr96 s[{{[0-9]+}}:{{[0-9]+}}]
; CI: NumSgprs: 84
; VI-NOBUG: NumSgprs: 86
; VI-BUG: NumSgprs: 96
; GCN: NumVgprs: 50
define amdgpu_kernel void @count_use_sgpr96_external_call()  {
entry:
  tail call void asm sideeffect "; sgpr96 $0", "s"(<3 x i32> <i32 10, i32 11, i32 12>) nounwind noinline norecurse
  call void @external()
  ret void
}

; Make sure there's no assert when a sgpr160 is used.
; GCN-LABEL: {{^}}count_use_sgpr160_external_call
; GCN: ; sgpr160 s[{{[0-9]+}}:{{[0-9]+}}]
; CI: NumSgprs: 84
; VI-NOBUG: NumSgprs: 86
; VI-BUG: NumSgprs: 96
; GCN: NumVgprs: 50
define amdgpu_kernel void @count_use_sgpr160_external_call()  {
entry:
  tail call void asm sideeffect "; sgpr160 $0", "s"(<5 x i32> <i32 10, i32 11, i32 12, i32 13, i32 14>) nounwind noinline norecurse
  call void @external()
  ret void
}

; Make sure there's no assert when a vgpr160 is used.
; GCN-LABEL: {{^}}count_use_vgpr160_external_call
; GCN: ; vgpr160 v[{{[0-9]+}}:{{[0-9]+}}]
; CI: NumSgprs: 84
; VI-NOBUG: NumSgprs: 86
; VI-BUG: NumSgprs: 96
; GCN: NumVgprs: 50
define amdgpu_kernel void @count_use_vgpr160_external_call()  {
entry:
  tail call void asm sideeffect "; vgpr160 $0", "v"(<5 x i32> <i32 10, i32 11, i32 12, i32 13, i32 14>) nounwind noinline norecurse
  call void @external()
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 CODE_OBJECT_VERSION}
