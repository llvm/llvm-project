; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,VI,GFX678,HAS-ATOMICS %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9,HAS-ATOMICS %s
; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX678,NO-ATOMICS %s
; RUN: llc -march=amdgcn -mcpu=hawaii -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX678,NO-ATOMICS %s

; GCN-LABEL: {{^}}lds_atomic_fadd_ret_f32:
; GFX678-DAG: s_mov_b32 m0
; GFX9-NOT: m0
; HAS-ATOMICS-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 4.0
; HAS-ATOMICS: ds_add_rtn_f32 v0, v0, [[K]]

; NO-ATOMICS: ds_read_b32
; NO-ATOMICS: v_add_f32
; NO-ATOMICS: ds_cmpst_rtn_b32
; NO-ATOMICS: s_cbranch_execnz
define float @lds_atomic_fadd_ret_f32(float addrspace(3)* %ptr) nounwind {
  %result = atomicrmw fadd float addrspace(3)* %ptr, float 4.0 seq_cst
  ret float %result
}

; GCN-LABEL: {{^}}lds_atomic_fadd_noret_f32:
; GFX678-DAG: s_mov_b32 m0
; GFX9-NOT: m0
; HAS-ATOMICS-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 4.0
; HAS-ATOMICS: ds_add_f32 v0, [[K]]
define void @lds_atomic_fadd_noret_f32(float addrspace(3)* %ptr) nounwind {
  %result = atomicrmw fadd float addrspace(3)* %ptr, float 4.0 seq_cst
  ret void
}

; GCN-LABEL: {{^}}lds_ds_fadd:
; VI-DAG: s_mov_b32 m0
; GFX9-NOT: m0
; HAS-ATOMICS-DAG: v_mov_b32_e32 [[V0:v[0-9]+]], 0x42280000
; HAS-ATOMICS: ds_add_rtn_f32 [[V2:v[0-9]+]], [[V1:v[0-9]+]], [[V0]] offset:32
; HAS-ATOMICS: ds_add_f32 [[V3:v[0-9]+]], [[V0]] offset:64
; HAS-ATOMICS: s_waitcnt lgkmcnt(0)
; HAS-ATOMICS: ds_add_rtn_f32 {{v[0-9]+}}, {{v[0-9]+}}, [[V2]]
define amdgpu_kernel void @lds_ds_fadd(float addrspace(1)* %out, float addrspace(3)* %ptrf, i32 %idx) {
  %idx.add = add nuw i32 %idx, 4
  %shl0 = shl i32 %idx.add, 3
  %shl1 = shl i32 %idx.add, 4
  %ptr0 = inttoptr i32 %shl0 to float addrspace(3)*
  %ptr1 = inttoptr i32 %shl1 to float addrspace(3)*
  %a1 = atomicrmw fadd float addrspace(3)* %ptr0, float 4.2e+1 seq_cst
  %a2 = atomicrmw fadd float addrspace(3)* %ptr1, float 4.2e+1 seq_cst
  %a3 = atomicrmw fadd float addrspace(3)* %ptrf, float %a1 seq_cst
  store float %a3, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}lds_ds_fadd_one_as:
; VI-DAG: s_mov_b32 m0
; GFX9-NOT: m0
; HAS-ATOMICS-DAG: v_mov_b32_e32 [[V0:v[0-9]+]], 0x42280000
; HAS-ATOMICS: ds_add_rtn_f32 [[V2:v[0-9]+]], [[V1:v[0-9]+]], [[V0]] offset:32
; HAS-ATOMICS: ds_add_f32 [[V3:v[0-9]+]], [[V0]] offset:64
; HAS-ATOMICS: s_waitcnt lgkmcnt(1)
; HAS-ATOMICS: ds_add_rtn_f32 {{v[0-9]+}}, {{v[0-9]+}}, [[V2]]
define amdgpu_kernel void @lds_ds_fadd_one_as(float addrspace(1)* %out, float addrspace(3)* %ptrf, i32 %idx) {
  %idx.add = add nuw i32 %idx, 4
  %shl0 = shl i32 %idx.add, 3
  %shl1 = shl i32 %idx.add, 4
  %ptr0 = inttoptr i32 %shl0 to float addrspace(3)*
  %ptr1 = inttoptr i32 %shl1 to float addrspace(3)*
  %a1 = atomicrmw fadd float addrspace(3)* %ptr0, float 4.2e+1 syncscope("one-as") seq_cst
  %a2 = atomicrmw fadd float addrspace(3)* %ptr1, float 4.2e+1 syncscope("one-as") seq_cst
  %a3 = atomicrmw fadd float addrspace(3)* %ptrf, float %a1 syncscope("one-as") seq_cst
  store float %a3, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_fadd_ret_f64:
; GCN: ds_read_b64
; GCN: v_add_f64
; GCN: ds_cmpst_rtn_b64
; GCN: s_cbranch_execnz
define double @lds_atomic_fadd_ret_f64(double addrspace(3)* %ptr) nounwind {
  %result = atomicrmw fadd double addrspace(3)* %ptr, double 4.0 seq_cst
  ret double %result
}

; GCN-LABEL: {{^}}lds_atomic_fadd_noret_f64:
; GCN: ds_read_b64
; GCN: v_add_f64
; GCN: ds_cmpst_rtn_b64
; GCN: s_cbranch_execnz
define void @lds_atomic_fadd_noret_f64(double addrspace(3)* %ptr) nounwind {
  %result = atomicrmw fadd double addrspace(3)* %ptr, double 4.0 seq_cst
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_fsub_ret_f32:
; GCN: ds_read_b32
; GCN: v_sub_f32
; GCN: ds_cmpst_rtn_b32
; GCN: s_cbranch_execnz
define float @lds_atomic_fsub_ret_f32(float addrspace(3)* %ptr, float %val) nounwind {
  %result = atomicrmw fsub float addrspace(3)* %ptr, float %val seq_cst
  ret float %result
}

; GCN-LABEL: {{^}}lds_atomic_fsub_noret_f32:
; GCN: ds_read_b32
; GCN: v_sub_f32
; GCN: ds_cmpst_rtn_b32
define void @lds_atomic_fsub_noret_f32(float addrspace(3)* %ptr, float %val) nounwind {
  %result = atomicrmw fsub float addrspace(3)* %ptr, float %val seq_cst
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_fsub_ret_f64:
; GCN: ds_read_b64
; GCN: v_add_f64 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, -v{{\[[0-9]+:[0-9]+\]}}
; GCN: ds_cmpst_rtn_b64

define double @lds_atomic_fsub_ret_f64(double addrspace(3)* %ptr, double %val) nounwind {
  %result = atomicrmw fsub double addrspace(3)* %ptr, double %val seq_cst
  ret double %result
}

; GCN-LABEL: {{^}}lds_atomic_fsub_noret_f64:
; GCN: ds_read_b64
; GCN: v_add_f64 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, -v{{\[[0-9]+:[0-9]+\]}}
; GCN: ds_cmpst_rtn_b64
; GCN: s_cbranch_execnz
define void @lds_atomic_fsub_noret_f64(double addrspace(3)* %ptr, double %val) nounwind {
  %result = atomicrmw fsub double addrspace(3)* %ptr, double %val seq_cst
  ret void
}
