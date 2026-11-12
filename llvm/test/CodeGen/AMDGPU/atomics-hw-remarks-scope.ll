; RUN: llc -mtriple=amdgpu9.42 -pass-remarks='si-lower|atomic-expand' -filetype=null %s 2>&1 | \
; RUN:   FileCheck %s --check-prefix=GFX942 --implicit-check-not=remark:
; RUN: llc -mtriple=amdgpu12.50 -pass-remarks='si-lower|atomic-expand' -filetype=null %s 2>&1 | \
; RUN:   FileCheck %s --check-prefix=GFX1250 --implicit-check-not=remark:

; GFX942: remark: <unknown>:0:0: hardware instruction generated for atomic fadd at agent scope since fine-grained remote memory atomics work below system scope
; GFX1250: remark: <unknown>:0:0: hardware instruction generated for atomic fadd at agent scope since fine-grained remote memory atomics work below system scope
define void @global_atomicrmw_fadd_f32_agent(ptr addrspace(1) %ptr, float %val) {
  %ret = atomicrmw fadd ptr addrspace(1) %ptr, float %val syncscope("agent") monotonic, align 4
  ret void
}

; GFX942: remark: <unknown>:0:0: A compare and swap loop was generated for an atomic fadd operation at system memory scope
; GFX1250: remark: <unknown>:0:0: hardware instruction generated for atomic fadd at system scope since system scope atomics are emulated in hardware
define void @global_atomicrmw_fadd_f32_system(ptr addrspace(1) %ptr, float %val) {
  %ret = atomicrmw fadd ptr addrspace(1) %ptr, float %val monotonic, align 4
  ret void
}

; GFX942: remark: <unknown>:0:0: hardware instruction generated for atomic fadd at system scope since memory is not remote (!amdgpu.no.remote.memory)
; GFX1250: remark: <unknown>:0:0: hardware instruction generated for atomic fadd at system scope since memory is not remote (!amdgpu.no.remote.memory)
define void @global_atomicrmw_fadd_f32_system__amdgpu_no_remote_memory(ptr addrspace(1) %ptr, float %val) {
  %ret = atomicrmw fadd ptr addrspace(1) %ptr, float %val monotonic, align 4, !amdgpu.no.remote.memory !0
  ret void
}

; GFX942: remark: <unknown>:0:0: hardware instruction generated for atomic fadd at system scope since memory is not fine-grained (!amdgpu.no.fine.grained.memory)
; GFX1250: remark: <unknown>:0:0: hardware instruction generated for atomic fadd at system scope since system scope atomics are emulated in hardware
define void @global_atomicrmw_fadd_f32_system__amdgpu_no_fine_grained_memory(ptr addrspace(1) %ptr, float %val) {
  %ret = atomicrmw fadd ptr addrspace(1) %ptr, float %val monotonic, align 4, !amdgpu.no.fine.grained.memory !0
  ret void
}

; GFX942: remark: <unknown>:0:0: hardware instruction generated for atomic fmax at agent scope since fine-grained remote memory atomics work below system scope
; GFX1250: remark: <unknown>:0:0: hardware instruction generated for atomic fmax at agent scope since fine-grained remote memory atomics work below system scope
define void @global_atomicrmw_fmax_f64_agent(ptr addrspace(1) %ptr, double %val) {
  %ret = atomicrmw fmax ptr addrspace(1) %ptr, double %val syncscope("agent") monotonic, align 8
  ret void
}

!0 = !{}
