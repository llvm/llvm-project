; RUN: llc -mtriple=amdgcn -mcpu=gfx90a -verify-machineinstrs --pass-remarks=si-lower \
; RUN:      %s -o - 2>&1 | FileCheck %s --check-prefix=GFX90A-HW

; GFX90A-HW: Hardware instruction generated for atomic fadd operation at memory scope agent due to an unsafe request.
; GFX90A-HW: Hardware instruction generated for atomic fadd operation at memory scope workgroup due to an unsafe request.
; GFX90A-HW: Hardware instruction generated for atomic fadd operation at memory scope wavefront due to an unsafe request.
; GFX90A-HW: Hardware instruction generated for atomic fadd operation at memory scope singlethread due to an unsafe request.
; GFX90A-HW: Hardware instruction generated for atomic fadd operation at memory scope agent-one-as due to an unsafe request.
; GFX90A-HW: Hardware instruction generated for atomic fadd operation at memory scope workgroup-one-as due to an unsafe request.
; GFX90A-HW: Hardware instruction generated for atomic fadd operation at memory scope wavefront-one-as due to an unsafe request.
; GFX90A-HW: Hardware instruction generated for atomic fadd operation at memory scope singlethread-one-as due to an unsafe request.

; GFX90A-HW-LABEL: atomic_add_unsafe_hw:
; GFX90A-HW:    ds_add_f64 v2, v[0:1]
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

attributes #0 = { "denormal-fp-math"="preserve-sign,preserve-sign" }

!0 = !{}
