; RUN: llc -march=amdgcn -mtriple=amdgcn---amdgiz -amdgpu-atomic-optimizations=true -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX7LESS %s
; RUN: llc -march=amdgcn -mtriple=amdgcn---amdgiz -mcpu=tonga -mattr=-flat-for-global -amdgpu-atomic-optimizations=true -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX8MORE %s
; RUN: llc -march=amdgcn -mtriple=amdgcn---amdgiz -mcpu=gfx900 -mattr=-flat-for-global -amdgpu-atomic-optimizations=true -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX8MORE %s

declare i32 @llvm.amdgcn.workitem.id.x()

@local_var32 = addrspace(3) global i32 undef, align 4
@local_var64 = addrspace(3) global i64 undef, align 8

; Show that what the atomic optimization pass will do for local pointers.

; GCN-LABEL: add_i32_constant:
; GCN: s_mov_b64 s{{\[}}[[exec_lo:[0-9]+]]:[[exec_hi:[0-9]+]]{{\]}}, exec
; GCN: v_mbcnt_lo_u32_b32{{(_e[0-9]+)?}} v[[mbcnt_lo:[0-9]+]], s[[exec_lo]], 0
; GCN: v_mbcnt_hi_u32_b32{{(_e[0-9]+)?}} v[[mbcnt_hi:[0-9]+]], s[[exec_hi]], v[[mbcnt_lo]]
; GCN: v_cmp_eq_u32{{(_e[0-9]+)?}} vcc, 0, v[[mbcnt_hi]]
; GCN: s_bcnt1_i32_b64 s[[popcount:[0-9]+]], s{{\[}}[[exec_lo]]:[[exec_hi]]{{\]}}
; GCN: v_mul_u32_u24{{(_e[0-9]+)?}} v[[value:[0-9]+]], s[[popcount]], 5
; GCN: ds_add_rtn_u32 v{{[0-9]+}}, v{{[0-9]+}}, v[[value]]
define amdgpu_kernel void @add_i32_constant(i32 addrspace(1)* %out) {
entry:
  %old = atomicrmw add i32 addrspace(3)* @local_var32, i32 5 acq_rel
  store i32 %old, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: add_i32_uniform:
; GCN: s_mov_b64 s{{\[}}[[exec_lo:[0-9]+]]:[[exec_hi:[0-9]+]]{{\]}}, exec
; GCN: v_mbcnt_lo_u32_b32{{(_e[0-9]+)?}} v[[mbcnt_lo:[0-9]+]], s[[exec_lo]], 0
; GCN: v_mbcnt_hi_u32_b32{{(_e[0-9]+)?}} v[[mbcnt_hi:[0-9]+]], s[[exec_hi]], v[[mbcnt_lo]]
; GCN: v_cmp_eq_u32{{(_e[0-9]+)?}} vcc, 0, v[[mbcnt_hi]]
; GCN: s_bcnt1_i32_b64 s[[popcount:[0-9]+]], s{{\[}}[[exec_lo]]:[[exec_hi]]{{\]}}
; GCN: s_mul_i32 s[[scalar_value:[0-9]+]], s{{[0-9]+}}, s[[popcount]]
; GCN: v_mov_b32{{(_e[0-9]+)?}} v[[value:[0-9]+]], s[[scalar_value]]
; GCN: ds_add_rtn_u32 v{{[0-9]+}}, v{{[0-9]+}}, v[[value]]
define amdgpu_kernel void @add_i32_uniform(i32 addrspace(1)* %out, i32 %additive) {
entry:
  %old = atomicrmw add i32 addrspace(3)* @local_var32, i32 %additive acq_rel
  store i32 %old, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: add_i32_varying:
; GFX7LESS-NOT: v_mbcnt_lo_u32_b32
; GFX7LESS-NOT: v_mbcnt_hi_u32_b32
; GFX7LESS-NOT: s_bcnt1_i32_b64
; GFX7LESS: ds_add_rtn_u32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GFX8MORE: v_readlane_b32 s[[scalar_value:[0-9]+]], v{{[0-9]+}}, 63
; GFX8MORE: v_mov_b32{{(_e[0-9]+)?}} v[[value:[0-9]+]], s[[scalar_value]]
; GFX8MORE: ds_add_rtn_u32 v{{[0-9]+}}, v{{[0-9]+}}, v[[value]]
define amdgpu_kernel void @add_i32_varying(i32 addrspace(1)* %out) {
entry:
  %lane = call i32 @llvm.amdgcn.workitem.id.x()
  %old = atomicrmw add i32 addrspace(3)* @local_var32, i32 %lane acq_rel
  store i32 %old, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: add_i64_constant:
; GCN: s_mov_b64 s{{\[}}[[exec_lo:[0-9]+]]:[[exec_hi:[0-9]+]]{{\]}}, exec
; GCN: v_mbcnt_lo_u32_b32{{(_e[0-9]+)?}} v[[mbcnt_lo:[0-9]+]], s[[exec_lo]], 0
; GCN: v_mbcnt_hi_u32_b32{{(_e[0-9]+)?}} v[[mbcnt_hi:[0-9]+]], s[[exec_hi]], v[[mbcnt_lo]]
; GCN: v_cmp_eq_u32{{(_e[0-9]+)?}} vcc, 0, v[[mbcnt_hi]]
; GCN: s_bcnt1_i32_b64 s[[popcount:[0-9]+]], s{{\[}}[[exec_lo]]:[[exec_hi]]{{\]}}
; GCN: v_mul_hi_u32_u24{{(_e[0-9]+)?}} v[[value_hi:[0-9]+]], s[[popcount]], 5
; GCN: v_mul_u32_u24{{(_e[0-9]+)?}} v[[value_lo:[0-9]+]], s[[popcount]], 5
; GCN: ds_add_rtn_u64 v{{\[}}{{[0-9]+}}:{{[0-9]+}}{{\]}}, v{{[0-9]+}}, v{{\[}}[[value_lo]]:[[value_hi]]{{\]}}
define amdgpu_kernel void @add_i64_constant(i64 addrspace(1)* %out) {
entry:
  %old = atomicrmw add i64 addrspace(3)* @local_var64, i64 5 acq_rel
  store i64 %old, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: add_i64_uniform:
; GCN: s_mov_b64 s{{\[}}[[exec_lo:[0-9]+]]:[[exec_hi:[0-9]+]]{{\]}}, exec
; GCN: v_mbcnt_lo_u32_b32{{(_e[0-9]+)?}} v[[mbcnt_lo:[0-9]+]], s[[exec_lo]], 0
; GCN: v_mbcnt_hi_u32_b32{{(_e[0-9]+)?}} v[[mbcnt_hi:[0-9]+]], s[[exec_hi]], v[[mbcnt_lo]]
; GCN: v_cmp_eq_u32{{(_e[0-9]+)?}} vcc, 0, v[[mbcnt_hi]]
; GCN: s_bcnt1_i32_b64 s{{[0-9]+}}, s{{\[}}[[exec_lo]]:[[exec_hi]]{{\]}}
; GCN: ds_add_rtn_u64 v{{\[}}{{[0-9]+}}:{{[0-9]+}}{{\]}}, v{{[0-9]+}}, v{{\[}}{{[0-9]+}}:{{[0-9]+}}{{\]}}
define amdgpu_kernel void @add_i64_uniform(i64 addrspace(1)* %out, i64 %additive) {
entry:
  %old = atomicrmw add i64 addrspace(3)* @local_var64, i64 %additive acq_rel
  store i64 %old, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: add_i64_varying:
; GCN-NOT: v_mbcnt_lo_u32_b32
; GCN-NOT: v_mbcnt_hi_u32_b32
; GCN-NOT: s_bcnt1_i32_b64
; GCN: ds_add_rtn_u64 v{{\[}}{{[0-9]+}}:{{[0-9]+}}{{\]}}, v{{[0-9]+}}, v{{\[}}{{[0-9]+}}:{{[0-9]+}}{{\]}}
define amdgpu_kernel void @add_i64_varying(i64 addrspace(1)* %out) {
entry:
  %lane = call i32 @llvm.amdgcn.workitem.id.x()
  %zext = zext i32 %lane to i64
  %old = atomicrmw add i64 addrspace(3)* @local_var64, i64 %zext acq_rel
  store i64 %old, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: sub_i32_constant:
; GCN: s_mov_b64 s{{\[}}[[exec_lo:[0-9]+]]:[[exec_hi:[0-9]+]]{{\]}}, exec
; GCN: v_mbcnt_lo_u32_b32{{(_e[0-9]+)?}} v[[mbcnt_lo:[0-9]+]], s[[exec_lo]], 0
; GCN: v_mbcnt_hi_u32_b32{{(_e[0-9]+)?}} v[[mbcnt_hi:[0-9]+]], s[[exec_hi]], v[[mbcnt_lo]]
; GCN: v_cmp_eq_u32{{(_e[0-9]+)?}} vcc, 0, v[[mbcnt_hi]]
; GCN: s_bcnt1_i32_b64 s[[popcount:[0-9]+]], s{{\[}}[[exec_lo]]:[[exec_hi]]{{\]}}
; GCN: v_mul_u32_u24{{(_e[0-9]+)?}} v[[value:[0-9]+]], s[[popcount]], 5
; GCN: ds_sub_rtn_u32 v{{[0-9]+}}, v{{[0-9]+}}, v[[value]]
define amdgpu_kernel void @sub_i32_constant(i32 addrspace(1)* %out) {
entry:
  %old = atomicrmw sub i32 addrspace(3)* @local_var32, i32 5 acq_rel
  store i32 %old, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: sub_i32_uniform:
; GCN: s_mov_b64 s{{\[}}[[exec_lo:[0-9]+]]:[[exec_hi:[0-9]+]]{{\]}}, exec
; GCN: v_mbcnt_lo_u32_b32{{(_e[0-9]+)?}} v[[mbcnt_lo:[0-9]+]], s[[exec_lo]], 0
; GCN: v_mbcnt_hi_u32_b32{{(_e[0-9]+)?}} v[[mbcnt_hi:[0-9]+]], s[[exec_hi]], v[[mbcnt_lo]]
; GCN: v_cmp_eq_u32{{(_e[0-9]+)?}} vcc, 0, v[[mbcnt_hi]]
; GCN: s_bcnt1_i32_b64 s[[popcount:[0-9]+]], s{{\[}}[[exec_lo]]:[[exec_hi]]{{\]}}
; GCN: s_mul_i32 s[[scalar_value:[0-9]+]], s{{[0-9]+}}, s[[popcount]]
; GCN: v_mov_b32{{(_e[0-9]+)?}} v[[value:[0-9]+]], s[[scalar_value]]
; GCN: ds_sub_rtn_u32 v{{[0-9]+}}, v{{[0-9]+}}, v[[value]]
define amdgpu_kernel void @sub_i32_uniform(i32 addrspace(1)* %out, i32 %subitive) {
entry:
  %old = atomicrmw sub i32 addrspace(3)* @local_var32, i32 %subitive acq_rel
  store i32 %old, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: sub_i32_varying:
; GFX7LESS-NOT: v_mbcnt_lo_u32_b32
; GFX7LESS-NOT: v_mbcnt_hi_u32_b32
; GFX7LESS-NOT: s_bcnt1_i32_b64
; GFX7LESS: ds_sub_rtn_u32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GFX8MORE: v_readlane_b32 s[[scalar_value:[0-9]+]], v{{[0-9]+}}, 63
; GFX8MORE: v_mov_b32{{(_e[0-9]+)?}} v[[value:[0-9]+]], s[[scalar_value]]
; GFX8MORE: ds_sub_rtn_u32 v{{[0-9]+}}, v{{[0-9]+}}, v[[value]]
define amdgpu_kernel void @sub_i32_varying(i32 addrspace(1)* %out) {
entry:
  %lane = call i32 @llvm.amdgcn.workitem.id.x()
  %old = atomicrmw sub i32 addrspace(3)* @local_var32, i32 %lane acq_rel
  store i32 %old, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: sub_i64_constant:
; GCN: s_mov_b64 s{{\[}}[[exec_lo:[0-9]+]]:[[exec_hi:[0-9]+]]{{\]}}, exec
; GCN: v_mbcnt_lo_u32_b32{{(_e[0-9]+)?}} v[[mbcnt_lo:[0-9]+]], s[[exec_lo]], 0
; GCN: v_mbcnt_hi_u32_b32{{(_e[0-9]+)?}} v[[mbcnt_hi:[0-9]+]], s[[exec_hi]], v[[mbcnt_lo]]
; GCN: v_cmp_eq_u32{{(_e[0-9]+)?}} vcc, 0, v[[mbcnt_hi]]
; GCN: s_bcnt1_i32_b64 s[[popcount:[0-9]+]], s{{\[}}[[exec_lo]]:[[exec_hi]]{{\]}}
; GCN: v_mul_hi_u32_u24{{(_e[0-9]+)?}} v[[value_hi:[0-9]+]], s[[popcount]], 5
; GCN: v_mul_u32_u24{{(_e[0-9]+)?}} v[[value_lo:[0-9]+]], s[[popcount]], 5
; GCN: ds_sub_rtn_u64 v{{\[}}{{[0-9]+}}:{{[0-9]+}}{{\]}}, v{{[0-9]+}}, v{{\[}}[[value_lo]]:[[value_hi]]{{\]}}
define amdgpu_kernel void @sub_i64_constant(i64 addrspace(1)* %out) {
entry:
  %old = atomicrmw sub i64 addrspace(3)* @local_var64, i64 5 acq_rel
  store i64 %old, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: sub_i64_uniform:
; GCN: s_mov_b64 s{{\[}}[[exec_lo:[0-9]+]]:[[exec_hi:[0-9]+]]{{\]}}, exec
; GCN: v_mbcnt_lo_u32_b32{{(_e[0-9]+)?}} v[[mbcnt_lo:[0-9]+]], s[[exec_lo]], 0
; GCN: v_mbcnt_hi_u32_b32{{(_e[0-9]+)?}} v[[mbcnt_hi:[0-9]+]], s[[exec_hi]], v[[mbcnt_lo]]
; GCN: v_cmp_eq_u32{{(_e[0-9]+)?}} vcc, 0, v[[mbcnt_hi]]
; GCN: s_bcnt1_i32_b64 s{{[0-9]+}}, s{{\[}}[[exec_lo]]:[[exec_hi]]{{\]}}
; GCN: ds_sub_rtn_u64 v{{\[}}{{[0-9]+}}:{{[0-9]+}}{{\]}}, v{{[0-9]+}}, v{{\[}}{{[0-9]+}}:{{[0-9]+}}{{\]}}
define amdgpu_kernel void @sub_i64_uniform(i64 addrspace(1)* %out, i64 %subitive) {
entry:
  %old = atomicrmw sub i64 addrspace(3)* @local_var64, i64 %subitive acq_rel
  store i64 %old, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: sub_i64_varying:
; GCN-NOT: v_mbcnt_lo_u32_b32
; GCN-NOT: v_mbcnt_hi_u32_b32
; GCN-NOT: s_bcnt1_i32_b64
; GCN: ds_sub_rtn_u64 v{{\[}}{{[0-9]+}}:{{[0-9]+}}{{\]}}, v{{[0-9]+}}, v{{\[}}{{[0-9]+}}:{{[0-9]+}}{{\]}}
define amdgpu_kernel void @sub_i64_varying(i64 addrspace(1)* %out) {
entry:
  %lane = call i32 @llvm.amdgcn.workitem.id.x()
  %zext = zext i32 %lane to i64
  %old = atomicrmw sub i64 addrspace(3)* @local_var64, i64 %zext acq_rel
  store i64 %old, i64 addrspace(1)* %out
  ret void
}
