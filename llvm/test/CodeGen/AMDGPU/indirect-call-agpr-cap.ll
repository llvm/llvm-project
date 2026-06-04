; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 < %s | FileCheck %s --check-prefixes=CHECK

declare void @extern_fn() #0          ; forces HasIndirectCall path
define void @big_vgpr_user() #1 {
    call void asm sideeffect "","~{a125}"() #1
    ret void
}
define internal void @small_callee() {
    ret void
}

define amdgpu_kernel void @kernel_indirect_attributed() #0 {
    
    call void @extern_fn()
    ret void
}

; CHECK: .set .Lkernel_indirect_attributed.num_vgpr, min(64, max(32, amdgpu.max_num_vgpr))
; CHECK: .set .Lkernel_indirect_attributed.num_agpr, min(64, max(0, amdgpu.max_num_agpr))

define amdgpu_kernel void @kernel_indirect_unattributed() { 
    call void @extern_fn()
    ret void
}

; CHECK: .set .Lkernel_indirect_unattributed.num_vgpr, min(64, max(32, amdgpu.max_num_vgpr))
; CHECK: .set .Lkernel_indirect_unattributed.num_agpr, min(64, max(0, amdgpu.max_num_agpr))

define amdgpu_kernel void @kernel_direct_only() {
    call void @small_callee()
    ret void
}

; CHECK: .set .Lkernel_direct_only.num_vgpr, max({{[0-9]+}}, .Lsmall_callee.num_vgpr)


attributes #0 = {"amdgpu-flat-work-group-size"="1,1024"}
attributes #1 = {nounwind noinline norecurse}
attributes #2 = {nounwind noinline norecurse}

; CHECK: .set amdgpu.max_num_agpr, 126



; CHECK: .agpr_count: 64
; CHECK-LABEL: .symbol: kernel_indirect_attributed.kd
; CHECK: .vgpr_count: 96

; CHECK: .agpr_count: 64
; CHECK-LABEL:  .symbol: kernel_indirect_unattributed.kd
; CHECK: .vgpr_count: 96

; CHECK: .agpr_count: 0
; CHECK-LABEL: .symbol: kernel_direct_only.kd
; CHECK: .vgpr_count: 32

