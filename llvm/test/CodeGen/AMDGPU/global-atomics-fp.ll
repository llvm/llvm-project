; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX900 %s
; RUN: llc -march=amdgcn -mcpu=gfx908 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX908 %s

; GCN-LABEL: {{^}}global_atomic_fadd_ret_f32:
; GCN: [[LOOP:BB[0-9]+_[0-9]+]]
; GCN: v_add_f32_e32
; GCN: global_atomic_cmpswap
; GCN: s_andn2_b64 exec, exec,
; GCN-NEXT: s_cbranch_execnz [[LOOP]]
define amdgpu_kernel void @global_atomic_fadd_ret_f32(float addrspace(1)* %ptr) #0 {
  %result = atomicrmw fadd float addrspace(1)* %ptr, float 4.0 seq_cst
  store float %result, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}global_atomic_fadd_ret_f32_ieee:
; GCN: [[LOOP:BB[0-9]+_[0-9]+]]
; GCN: v_add_f32_e32
; GCN: global_atomic_cmpswap
; GCN: s_andn2_b64 exec, exec,
; GCN-NEXT: s_cbranch_execnz [[LOOP]]
define amdgpu_kernel void @global_atomic_fadd_ret_f32_ieee(float addrspace(1)* %ptr) {
  %result = atomicrmw fadd float addrspace(1)* %ptr, float 4.0 seq_cst
  store float %result, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}global_atomic_fadd_noret_f32:
; GFX900: [[LOOP:BB[0-9]+_[0-9]+]]
; GFX900: v_add_f32_e32
; GFX900: global_atomic_cmpswap
; GFX900: s_andn2_b64 exec, exec,
; GFX900-NEXT: s_cbranch_execnz [[LOOP]]

; GFX908-NOT: v_add_f32
; GFX908:     global_atomic_add_f32 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, off
; GFX908-NOT: s_cbranch_execnz
define amdgpu_kernel void @global_atomic_fadd_noret_f32(float addrspace(1)* %ptr) #0 {
  %result = atomicrmw fadd float addrspace(1)* %ptr, float 4.0 seq_cst
  ret void
}

; GCN-LABEL: {{^}}global_atomic_fadd_noret_f32_ieee:
; GCN: global_atomic_cmpswap
define amdgpu_kernel void @global_atomic_fadd_noret_f32_ieee(float addrspace(1)* %ptr) {
  %result = atomicrmw fadd float addrspace(1)* %ptr, float 4.0 seq_cst
  ret void
}

; Make sure this artificially selects with an incorrect subtarget, but the feature set.
; GCN-LABEL: {{^}}global_atomic_fadd_ret_f32_wrong_subtarget:
define amdgpu_kernel void @global_atomic_fadd_ret_f32_wrong_subtarget(float addrspace(1)* %ptr) #1 {
  %result = atomicrmw fadd float addrspace(1)* %ptr, float 4.0 seq_cst
  store float %result, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}global_atomic_fadd_noret_f32_wrong_subtarget:
define amdgpu_kernel void @global_atomic_fadd_noret_f32_wrong_subtarget(float addrspace(1)* %ptr) #1 {
  %result = atomicrmw fadd float addrspace(1)* %ptr, float 4.0 seq_cst
  ret void
}

attributes #0 = { "denormal-fp-math-f32"="preserve-sign,preserve-sign"}
attributes #1 = { "denormal-fp-math-f32"="preserve-sign,preserve-sign" "target-cpu"="gfx803" "target-features"="+atomic-fadd-insts" }
