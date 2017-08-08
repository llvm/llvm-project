; RUN: llc < %s -march=amdgcn -mcpu=bonaire -verify-machineinstrs | FileCheck --check-prefix=GCN %s
; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck --check-prefix=GCN %s

; GCN-LABEL: {{^}}inline_reg_constraints:
; GCN: flat_load_dword v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}]
; GCN: flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
; GCN: flat_load_dwordx4 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
; GCN: s_load_dword s{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}]
; GCN: s_load_dwordx2 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}]
; GCN: s_load_dwordx4 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}]
; GCN: s_load_dwordx8 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}]

define amdgpu_kernel void @inline_reg_constraints(i32 addrspace(1)* %ptr) {
entry:
  %v32 = tail call i32 asm sideeffect "flat_load_dword   $0, $1", "=v,v"(i32 addrspace(1)* %ptr)
  %v64 = tail call <2 x i32> asm sideeffect "flat_load_dwordx2 $0, $1", "=v,v"(i32 addrspace(1)* %ptr)
  %v128 = tail call <4 x i32> asm sideeffect "flat_load_dwordx4 $0, $1", "=v,v"(i32 addrspace(1)* %ptr)
  %s32 =  tail call i32 asm sideeffect "s_load_dword $0, $1", "=s,s"(i32 addrspace(1)* %ptr)
  %s64 =  tail call <2 x i32> asm sideeffect "s_load_dwordx2 $0, $1", "=s,s"(i32 addrspace(1)* %ptr)
  %s128 =  tail call <4 x i32> asm sideeffect "s_load_dwordx4 $0, $1", "=s,s"(i32 addrspace(1)* %ptr)
  %s256 =  tail call <8 x i32> asm sideeffect "s_load_dwordx8 $0, $1", "=s,s"(i32 addrspace(1)* %ptr)
  ret void
}

; FIXME: Should be able to avoid copy
; GCN-LABEL: {{^}}inline_sreg_constraint_m0:
; GCN: s_mov_b32 m0, -1
; GCN: s_mov_b32 [[COPY_M0:s[0-9]+]], m0
; GCN: ; use [[COPY_M0]]
define amdgpu_kernel void @inline_sreg_constraint_m0() {
  %m0 = tail call i32 asm sideeffect "s_mov_b32 m0, -1", "={M0}"()
  tail call void asm sideeffect "; use $0", "s"(i32 %m0)
  ret void
}

; GCN-LABEL: {{^}}inline_sreg_constraint_imm_i32:
; GCN: s_mov_b32 [[REG:s[0-9]+]], 32
; GCN: ; use [[REG]]
define amdgpu_kernel void @inline_sreg_constraint_imm_i32() {
  tail call void asm sideeffect "; use $0", "s"(i32 32)
  ret void
}

; GCN-LABEL: {{^}}inline_sreg_constraint_imm_f32:
; GCN: s_mov_b32 [[REG:s[0-9]+]], 1.0
; GCN: ; use [[REG]]
define amdgpu_kernel void @inline_sreg_constraint_imm_f32() {
  tail call void asm sideeffect "; use $0", "s"(float 1.0)
  ret void
}

; FIXME: Should be able to use s_mov_b64
; GCN-LABEL: {{^}}inline_sreg_constraint_imm_i64:
; GCN-DAG: s_mov_b32 s[[REG_LO:[0-9]+]], -4{{$}}
; GCN-DAG: s_mov_b32 s[[REG_HI:[0-9]+]], -1{{$}}
; GCN: ; use s{{\[}}[[REG_LO]]:[[REG_HI]]{{\]}}
define amdgpu_kernel void @inline_sreg_constraint_imm_i64() {
  tail call void asm sideeffect "; use $0", "s"(i64 -4)
  ret void
}

; GCN-LABEL: {{^}}inline_sreg_constraint_imm_f64:
; GCN-DAG: s_mov_b32 s[[REG_LO:[0-9]+]], 0{{$}}
; GCN-DAG: s_mov_b32 s[[REG_HI:[0-9]+]], 0x3ff00000{{$}}
; GCN: ; use s{{\[}}[[REG_LO]]:[[REG_HI]]{{\]}}
define amdgpu_kernel void @inline_sreg_constraint_imm_f64() {
  tail call void asm sideeffect "; use $0", "s"(double 1.0)
  ret void
}
