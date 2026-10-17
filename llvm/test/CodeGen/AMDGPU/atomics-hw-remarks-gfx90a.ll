; RUN: llc -mtriple=amdgpu9.0a --pass-remarks='si-lower|atomic-expand' %s -o - 2>&1 | FileCheck --check-prefix=GFX90A-HW %s

; GFX90A-HW: hardware instruction generated for atomic fadd at agent scope since memory is not fine-grained (!amdgpu.no.fine.grained.memory), and the floating-point environment flushes denormals
; GFX90A-HW: hardware instruction generated for atomic fadd at workgroup scope since memory is not fine-grained (!amdgpu.no.fine.grained.memory), and the floating-point environment flushes denormals
; GFX90A-HW: hardware instruction generated for atomic fadd at wavefront scope since memory is not fine-grained (!amdgpu.no.fine.grained.memory), and the floating-point environment flushes denormals
; GFX90A-HW: hardware instruction generated for atomic fadd at singlethread scope since memory is not fine-grained (!amdgpu.no.fine.grained.memory), and the floating-point environment flushes denormals
; GFX90A-HW: hardware instruction generated for atomic fadd at agent-one-as scope since memory is not fine-grained (!amdgpu.no.fine.grained.memory), and the floating-point environment flushes denormals
; GFX90A-HW: hardware instruction generated for atomic fadd at workgroup-one-as scope since memory is not fine-grained (!amdgpu.no.fine.grained.memory), and the floating-point environment flushes denormals
; GFX90A-HW: hardware instruction generated for atomic fadd at wavefront-one-as scope since memory is not fine-grained (!amdgpu.no.fine.grained.memory), and the floating-point environment flushes denormals
; GFX90A-HW: hardware instruction generated for atomic fadd at singlethread-one-as scope since memory is not fine-grained (!amdgpu.no.fine.grained.memory), and the floating-point environment flushes denormals

; GFX90A-HW-LABEL: atomic_add_unsafe_hw:
; GFX90A-HW:    ds_add_f64 v0, v[2:3]
; GFX90A-HW:    s_endpgm
define amdgpu_kernel void @atomic_add_unsafe_hw(ptr addrspace(3) %ptr) #0 {
main_body:
  %ret = atomicrmw fadd ptr addrspace(3) %ptr, double 4.0 seq_cst
  ret void
}

; GFX90A-HW-LABEL: atomic_add_unsafe_hw_agent:
; GFX90A-HW:    global_atomic_add_f32 v0, v1, s[2:3]
; GFX90A-HW:    s_endpgm
define amdgpu_kernel void @atomic_add_unsafe_hw_agent(ptr addrspace(1) %ptr, float %val) #0 {
main_body:
  %ret = atomicrmw fadd ptr addrspace(1) %ptr, float %val syncscope("agent") monotonic, align 4, !amdgpu.no.fine.grained.memory !0
  ret void
}

; GFX90A-HW-LABEL: atomic_add_unsafe_hw_wg:
; GFX90A-HW:    global_atomic_add_f32 v0, v1, s[2:3]
; GFX90A-HW:    s_endpgm
define amdgpu_kernel void @atomic_add_unsafe_hw_wg(ptr addrspace(1) %ptr, float %val) #0 {
main_body:
  %ret = atomicrmw fadd ptr addrspace(1) %ptr, float %val syncscope("workgroup") monotonic, align 4, !amdgpu.no.fine.grained.memory !0
  ret void
}

; GFX90A-HW-LABEL: atomic_add_unsafe_hw_wavefront:
; GFX90A-HW:    global_atomic_add_f32 v0, v1, s[2:3]
; GFX90A-HW:    s_endpgm
define amdgpu_kernel void @atomic_add_unsafe_hw_wavefront(ptr addrspace(1) %ptr, float %val) #0 {
main_body:
  %ret = atomicrmw fadd ptr addrspace(1) %ptr, float %val syncscope("wavefront") monotonic, align 4, !amdgpu.no.fine.grained.memory !0
  ret void
}

; GFX90A-HW-LABEL: atomic_add_unsafe_hw_single_thread:
; GFX90A-HW:    global_atomic_add_f32 v0, v1, s[2:3]
; GFX90A-HW:    s_endpgm
define amdgpu_kernel void @atomic_add_unsafe_hw_single_thread(ptr addrspace(1) %ptr, float %val) #0 {
main_body:
  %ret = atomicrmw fadd ptr addrspace(1) %ptr, float %val syncscope("singlethread") monotonic, align 4, !amdgpu.no.fine.grained.memory !0
  ret void
}

; GFX90A-HW-LABEL: atomic_add_unsafe_hw_aoa:
; GFX90A-HW:    global_atomic_add_f32 v0, v1, s[2:3]
; GFX90A-HW:    s_endpgm
define amdgpu_kernel void @atomic_add_unsafe_hw_aoa(ptr addrspace(1) %ptr, float %val) #0 {
main_body:
  %ret = atomicrmw fadd ptr addrspace(1) %ptr, float %val syncscope("agent-one-as") monotonic, align 4, !amdgpu.no.fine.grained.memory !0
  ret void
}

; GFX90A-HW-LABEL: atomic_add_unsafe_hw_wgoa:
; GFX90A-HW:    global_atomic_add_f32 v0, v1, s[2:3]
; GFX90A-HW:    s_endpgm
define amdgpu_kernel void @atomic_add_unsafe_hw_wgoa(ptr addrspace(1) %ptr, float %val) #0 {
main_body:
  %ret = atomicrmw fadd ptr addrspace(1) %ptr, float %val syncscope("workgroup-one-as") monotonic, align 4, !amdgpu.no.fine.grained.memory !0
  ret void
}

; GFX90A-HW-LABEL: atomic_add_unsafe_hw_wfoa:
; GFX90A-HW:    global_atomic_add_f32 v0, v1, s[2:3]
; GFX90A-HW:    s_endpgm
define amdgpu_kernel void @atomic_add_unsafe_hw_wfoa(ptr addrspace(1) %ptr, float %val) #0 {
main_body:
  %ret = atomicrmw fadd ptr addrspace(1) %ptr, float %val syncscope("wavefront-one-as") monotonic, align 4, !amdgpu.no.fine.grained.memory !0
  ret void
}

; GFX90A-HW-LABEL: atomic_add_unsafe_hw_stoa:
; GFX90A-HW:    global_atomic_add_f32 v0, v1, s[2:3]
; GFX90A-HW:    s_endpgm
define amdgpu_kernel void @atomic_add_unsafe_hw_stoa(ptr addrspace(1) %ptr, float %val) #0 {
main_body:
  %ret = atomicrmw fadd ptr addrspace(1) %ptr, float %val syncscope("singlethread-one-as") monotonic, align 4, !amdgpu.no.fine.grained.memory !0
  ret void
}

attributes #0 = { denormal_fpenv(preservesign) }

!0 = !{}
