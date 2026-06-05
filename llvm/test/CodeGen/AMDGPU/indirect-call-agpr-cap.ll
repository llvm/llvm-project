; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 < %s | FileCheck %s --check-prefixes=CHECK

; A kernel that makes an indirect call is assigned the module-wide maximum
; register usage (any function is a potential callee). When the kernel's own
; AGPR budget is lowered below the largestsibling function's explicit AGPR  
; usage, the reported/emitted AGPR count must be clamped to what the kernel 
; can actually allocate

declare void @extern_fn()   ; forces HasIndirectCall path
define void @big_agpr_user() #1 {
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

; CHECK: .set .Lkernel_indirect_attributed.num_agpr, min(40, max(0, amdgpu.max_num_agpr))

define amdgpu_kernel void @kernel_indirect_unattributed() { 
    call void @extern_fn()
    ret void
}

; CHECK: .set .Lkernel_indirect_unattributed.num_agpr, min(64, max(0, amdgpu.max_num_agpr))

define amdgpu_kernel void @kernel_direct_only() {
    call void @small_callee()
    ret void
}

; CHECK: .set .Lkernel_direct_only.num_vgpr, max(32, .Lsmall_callee.num_vgpr)


attributes #0 = {"amdgpu-num-vgpr"="40"}
attributes #1 = {nounwind noinline norecurse}

; CHECK: .set amdgpu.max_num_agpr, 126

; CHECK: .agpr_count: 40
; CHECK-LABEL: .symbol: kernel_indirect_attributed.kd
; CHECK: .vgpr_count: 72

; CHECK: .agpr_count: 64
; CHECK-LABEL:  .symbol: kernel_indirect_unattributed.kd
; CHECK: .vgpr_count: 96

; CHECK: .agpr_count: 0
; CHECK-LABEL: .symbol: kernel_direct_only.kd
; CHECK: .vgpr_count: 32

