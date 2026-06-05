; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 < %s | FileCheck %s --check-prefixes=CHECK

; A kernel that makes an indirect call is assigned the module-wide maximum
; register usage (any function is a potential callee). When the kernel's own
; VGPR budget is lowered below the largestsibling function's explicit VGPR  
; usage, the reported/emitted VGPR count must be clamped to what the kernel 
; can actually allocate

declare void @extern_fn()   ; forces HasIndirectCall path
define void @big_vgpr_user() #1 {
    call void asm sideeffect "","~{v125}"() #1
    ret void
}

: CHECK: .set .Lkernel_direct_only.has_indirect_call, or(0, .Lsmall_callee.has_indirect_call)
define internal void @small_callee() {
    ret void
}

; CHECK: .set .Lkernel_indirect_attributed.num_vgpr, min(40, max(32, amdgpu.max_num_vgpr))
; CHECK: .set .Lkernel_indirect_attributed.has_indirect_call, 1
define amdgpu_kernel void @kernel_indirect_attributed() #0 { 
    call void @extern_fn()
    ret void
}

; CHECK: .set .Lkernel_indirect_unattributed.num_vgpr, min(64, max(32, amdgpu.max_num_vgpr))
; CHECK: .set .Lkernel_indirect_unattributed.has_indirect_call, 1
define amdgpu_kernel void @kernel_indirect_unattributed() { 
    call void @extern_fn()
    ret void
}

; CHECK: .set .Lkernel_direct_only.num_vgpr, max(32, .Lsmall_callee.num_vgpr)
; CHECK: .set .Lkernel_direct_only.has_indirect_call, or(0, .Lsmall_callee.has_indirect_call)
define amdgpu_kernel void @kernel_direct_only() {
    call void @small_callee()
    ret void
}


attributes #0 = {"amdgpu-num-vgpr"="40"}
attributes #1 = {nounwind noinline norecurse "amdgpu-agpr-alloc"="0"}

; CHECK: .set amdgpu.max_num_vgpr, 126

; CHECK-LABEL: .symbol: kernel_indirect_attributed.kd
; CHECK: .vgpr_count: 40

; CHECK-LABEL:  .symbol: kernel_indirect_unattributed.kd
; CHECK: .vgpr_count: 64

; CHECK-LABEL: .symbol: kernel_direct_only.kd
; CHECK: .vgpr_count: 32

