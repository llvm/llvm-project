; RUN: llc -march=amdgcn -mtriple=amdgcn---amdgiz -amdgpu-atomic-optimizations=true -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX7LESS %s
; RUN: llc -march=amdgcn -mtriple=amdgcn---amdgiz -mcpu=tonga -mattr=-flat-for-global -amdgpu-atomic-optimizations=true -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX8MORE %s
; RUN: llc -march=amdgcn -mtriple=amdgcn---amdgiz -mcpu=gfx900 -mattr=-flat-for-global -amdgpu-atomic-optimizations=true -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX8MORE %s

declare i32 @llvm.amdgcn.workitem.id.x()
declare i32 @llvm.amdgcn.buffer.atomic.add(i32, <4 x i32>, i32, i32, i1)
declare i32 @llvm.amdgcn.buffer.atomic.sub(i32, <4 x i32>, i32, i32, i1)

; Show that what the atomic optimization pass will do for raw buffers.

; GCN-LABEL: add_i32_constant:
; GCN: s_mov_b64 s{{\[}}[[exec_lo:[0-9]+]]:[[exec_hi:[0-9]+]]{{\]}}, exec
; GCN: v_mbcnt_lo_u32_b32{{(_e[0-9]+)?}} v[[mbcnt_lo:[0-9]+]], s[[exec_lo]], 0
; GCN: v_mbcnt_hi_u32_b32{{(_e[0-9]+)?}} v[[mbcnt_hi:[0-9]+]], s[[exec_hi]], v[[mbcnt_lo]]
; GCN: v_cmp_eq_u32{{(_e[0-9]+)?}} vcc, 0, v[[mbcnt_hi]]
; GCN: s_bcnt1_i32_b64 s[[popcount:[0-9]+]], s{{\[}}[[exec_lo]]:[[exec_hi]]{{\]}}
; GCN: v_mul_u32_u24{{(_e[0-9]+)?}} v[[value:[0-9]+]], s[[popcount]], 5
; GCN: buffer_atomic_add v[[value]]
define amdgpu_kernel void @add_i32_constant(i32 addrspace(1)* %out, <4 x i32> %inout) {
entry:
  %old = call i32 @llvm.amdgcn.buffer.atomic.add(i32 5, <4 x i32> %inout, i32 0, i32 0, i1 0)
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
; GCN: buffer_atomic_add v[[value]]
define amdgpu_kernel void @add_i32_uniform(i32 addrspace(1)* %out, <4 x i32> %inout, i32 %additive) {
entry:
  %old = call i32 @llvm.amdgcn.buffer.atomic.add(i32 %additive, <4 x i32> %inout, i32 0, i32 0, i1 0)
  store i32 %old, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: add_i32_varying_vdata:
; GFX7LESS-NOT: v_mbcnt_lo_u32_b32
; GFX7LESS-NOT: v_mbcnt_hi_u32_b32
; GFX7LESS-NOT: s_bcnt1_i32_b64
; GFX7LESS: buffer_atomic_add v{{[0-9]+}}
; GFX8MORE: v_readlane_b32 s[[scalar_value:[0-9]+]], v{{[0-9]+}}, 63
; GFX8MORE: v_mov_b32{{(_e[0-9]+)?}} v[[value:[0-9]+]], s[[scalar_value]]
; GFX8MORE: buffer_atomic_add v[[value]]
define amdgpu_kernel void @add_i32_varying_vdata(i32 addrspace(1)* %out, <4 x i32> %inout) {
entry:
  %lane = call i32 @llvm.amdgcn.workitem.id.x()
  %old = call i32 @llvm.amdgcn.buffer.atomic.add(i32 %lane, <4 x i32> %inout, i32 0, i32 0, i1 0)
  store i32 %old, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: add_i32_varying_offset:
; GCN-NOT: v_mbcnt_lo_u32_b32
; GCN-NOT: v_mbcnt_hi_u32_b32
; GCN-NOT: s_bcnt1_i32_b64
; GCN: buffer_atomic_add v{{[0-9]+}}
define amdgpu_kernel void @add_i32_varying_offset(i32 addrspace(1)* %out, <4 x i32> %inout) {
entry:
  %lane = call i32 @llvm.amdgcn.workitem.id.x()
  %old = call i32 @llvm.amdgcn.buffer.atomic.add(i32 1, <4 x i32> %inout, i32 %lane, i32 0, i1 0)
  store i32 %old, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: sub_i32_constant:
; GCN: s_mov_b64 s{{\[}}[[exec_lo:[0-9]+]]:[[exec_hi:[0-9]+]]{{\]}}, exec
; GCN: v_mbcnt_lo_u32_b32{{(_e[0-9]+)?}} v[[mbcnt_lo:[0-9]+]], s[[exec_lo]], 0
; GCN: v_mbcnt_hi_u32_b32{{(_e[0-9]+)?}} v[[mbcnt_hi:[0-9]+]], s[[exec_hi]], v[[mbcnt_lo]]
; GCN: v_cmp_eq_u32{{(_e[0-9]+)?}} vcc, 0, v[[mbcnt_hi]]
; GCN: s_bcnt1_i32_b64 s[[popcount:[0-9]+]], s{{\[}}[[exec_lo]]:[[exec_hi]]{{\]}}
; GCN: v_mul_u32_u24{{(_e[0-9]+)?}} v[[value:[0-9]+]], s[[popcount]], 5
; GCN: buffer_atomic_sub v[[value]]
define amdgpu_kernel void @sub_i32_constant(i32 addrspace(1)* %out, <4 x i32> %inout) {
entry:
  %old = call i32 @llvm.amdgcn.buffer.atomic.sub(i32 5, <4 x i32> %inout, i32 0, i32 0, i1 0)
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
; GCN: buffer_atomic_sub v[[value]]
define amdgpu_kernel void @sub_i32_uniform(i32 addrspace(1)* %out, <4 x i32> %inout, i32 %subitive) {
entry:
  %old = call i32 @llvm.amdgcn.buffer.atomic.sub(i32 %subitive, <4 x i32> %inout, i32 0, i32 0, i1 0)
  store i32 %old, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: sub_i32_varying_vdata:
; GFX7LESS-NOT: v_mbcnt_lo_u32_b32
; GFX7LESS-NOT: v_mbcnt_hi_u32_b32
; GFX7LESS-NOT: s_bcnt1_i32_b64
; GFX7LESS: buffer_atomic_sub v{{[0-9]+}}
; GFX8MORE: v_readlane_b32 s[[scalar_value:[0-9]+]], v{{[0-9]+}}, 63
; GFX8MORE: v_mov_b32{{(_e[0-9]+)?}} v[[value:[0-9]+]], s[[scalar_value]]
; GFX8MORE: buffer_atomic_sub v[[value]]
define amdgpu_kernel void @sub_i32_varying_vdata(i32 addrspace(1)* %out, <4 x i32> %inout) {
entry:
  %lane = call i32 @llvm.amdgcn.workitem.id.x()
  %old = call i32 @llvm.amdgcn.buffer.atomic.sub(i32 %lane, <4 x i32> %inout, i32 0, i32 0, i1 0)
  store i32 %old, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: sub_i32_varying_offset:
; GCN-NOT: v_mbcnt_lo_u32_b32
; GCN-NOT: v_mbcnt_hi_u32_b32
; GCN-NOT: s_bcnt1_i32_b64
; GCN: buffer_atomic_sub v{{[0-9]+}}
define amdgpu_kernel void @sub_i32_varying_offset(i32 addrspace(1)* %out, <4 x i32> %inout) {
entry:
  %lane = call i32 @llvm.amdgcn.workitem.id.x()
  %old = call i32 @llvm.amdgcn.buffer.atomic.sub(i32 1, <4 x i32> %inout, i32 %lane, i32 0, i1 0)
  store i32 %old, i32 addrspace(1)* %out
  ret void
}
