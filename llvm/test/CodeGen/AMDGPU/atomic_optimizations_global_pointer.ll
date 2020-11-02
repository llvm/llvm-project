; RUN: llc -march=amdgcn -mtriple=amdgcn---amdgiz -amdgpu-atomic-optimizations=true -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GCN64,GFX7LESS %s
; RUN: llc -march=amdgcn -mtriple=amdgcn---amdgiz -mcpu=tonga -mattr=-flat-for-global -amdgpu-atomic-optimizations=true -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GCN64,GFX8MORE,GFX8MORE64,GFX89,DPPCOMB %s
; RUN: llc -march=amdgcn -mtriple=amdgcn---amdgiz -mcpu=gfx900 -mattr=-flat-for-global -amdgpu-atomic-optimizations=true -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GCN64,GFX8MORE,GFX8MORE64,GFX89,DPPCOMB %s
; RUN: llc -march=amdgcn -mtriple=amdgcn---amdgiz -mcpu=gfx1010 -mattr=-wavefrontsize32,+wavefrontsize64 -mattr=-flat-for-global -amdgpu-atomic-optimizations=true -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GCN64,GFX8MORE,GFX8MORE64,GFX10 %s
; RUN: llc -march=amdgcn -mtriple=amdgcn---amdgiz -mcpu=gfx1010 -mattr=+wavefrontsize32,-wavefrontsize64 -mattr=-flat-for-global -amdgpu-atomic-optimizations=true -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GCN32,GFX8MORE,GFX8MORE32,GFX10 %s

declare i32 @llvm.amdgcn.workitem.id.x()

; Show what the atomic optimization pass will do for global pointers.

; GCN-LABEL: add_i32_constant:
; GCN32: s_mov_b32 s[[exec_lo:[0-9]+]], exec_lo
; GCN64: s_mov_b64 s{{\[}}[[exec_lo:[0-9]+]]:[[exec_hi:[0-9]+]]{{\]}}, exec
; GCN: v_mbcnt_lo_u32_b32{{(_e[0-9]+)?}} v[[mbcnt:[0-9]+]], s[[exec_lo]], 0
; GCN64: v_mbcnt_hi_u32_b32{{(_e[0-9]+)?}} v[[mbcnt]], s[[exec_hi]], v[[mbcnt]]
; GCN: v_cmp_eq_u32{{(_e[0-9]+)?}} vcc{{(_lo)?}}, 0, v[[mbcnt]]
; GCN: s_and_saveexec_b{{32|64}} s[[exec:\[?[0-9:]+\]?]], vcc
; GCN32: s_bcnt1_i32_b32 s[[popcount:[0-9]+]], s[[exec_lo]]
; GCN64: s_bcnt1_i32_b64 s[[popcount:[0-9]+]], s{{\[}}[[exec_lo]]:[[exec_hi]]{{\]}}
; GCN: v_mul_u32_u24{{(_e[0-9]+)?}} v[[value:[0-9]+]], s[[popcount]], 5
; GCN: {{flat|buffer|global}}_atomic_add v[[value]]
define amdgpu_kernel void @add_i32_constant(i32 addrspace(1)* %out, i32 addrspace(1)* %inout) {
entry:
  %old = atomicrmw add i32 addrspace(1)* %inout, i32 5 acq_rel
  store i32 %old, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: add_i32_uniform:
; GCN32: s_mov_b32 s[[exec_lo:[0-9]+]], exec_lo
; GCN64: s_mov_b64 s{{\[}}[[exec_lo:[0-9]+]]:[[exec_hi:[0-9]+]]{{\]}}, exec
; GCN: v_mbcnt_lo_u32_b32{{(_e[0-9]+)?}} v[[mbcnt:[0-9]+]], s[[exec_lo]], 0
; GCN64: v_mbcnt_hi_u32_b32{{(_e[0-9]+)?}} v[[mbcnt]], s[[exec_hi]], v[[mbcnt]]
; GCN: v_cmp_eq_u32{{(_e[0-9]+)?}} vcc{{(_lo)?}}, 0, v[[mbcnt]]
; GCN: s_and_saveexec_b{{32|64}} s[[exec:\[?[0-9:]+\]?]], vcc
; GCN32: s_bcnt1_i32_b32 s[[popcount:[0-9]+]], s[[exec_lo]]
; GCN64: s_bcnt1_i32_b64 s[[popcount:[0-9]+]], s{{\[}}[[exec_lo]]:[[exec_hi]]{{\]}}
; GCN: s_mul_i32 s[[scalar_value:[0-9]+]], s{{[0-9]+}}, s[[popcount]]
; GCN: v_mov_b32{{(_e[0-9]+)?}} v[[value:[0-9]+]], s[[scalar_value]]
; GCN: {{flat|buffer|global}}_atomic_add v[[value]]
define amdgpu_kernel void @add_i32_uniform(i32 addrspace(1)* %out, i32 addrspace(1)* %inout, i32 %additive) {
entry:
  %old = atomicrmw add i32 addrspace(1)* %inout, i32 %additive acq_rel
  store i32 %old, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: add_i32_varying:
; GFX7LESS-NOT: v_mbcnt_lo_u32_b32
; GFX7LESS-NOT: v_mbcnt_hi_u32_b32
; GFX7LESS-NOT: s_bcnt1_i32_b64
; GFX7LESS: buffer_atomic_add v{{[0-9]+}}
; DPPCOMB: v_add_u32_dpp
; DPPCOMB: v_add_u32_dpp
; GFX8MORE32: v_readlane_b32 s[[scalar_value:[0-9]+]], v{{[0-9]+}}, 31
; GFX8MORE64: v_readlane_b32 s[[scalar_value:[0-9]+]], v{{[0-9]+}}, 63
; GFX89: v_mov_b32{{(_e[0-9]+)?}} v[[value:[0-9]+]], s[[scalar_value]]
; GFX10: s_mov_b32 s[[copy_value:[0-9]+]], s[[scalar_value]]
; GFX10: v_mov_b32{{(_e[0-9]+)?}} v[[value:[0-9]+]], s[[copy_value]]
; GFX8MORE: buffer_atomic_add v[[value]]
define amdgpu_kernel void @add_i32_varying(i32 addrspace(1)* %out, i32 addrspace(1)* %inout) {
entry:
  %lane = call i32 @llvm.amdgcn.workitem.id.x()
  %old = atomicrmw add i32 addrspace(1)* %inout, i32 %lane acq_rel
  store i32 %old, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: add_i64_constant:
; GCN32: s_mov_b32 s[[exec_lo:[0-9]+]], exec_lo
; GCN64: s_mov_b64 s{{\[}}[[exec_lo:[0-9]+]]:[[exec_hi:[0-9]+]]{{\]}}, exec
; GCN: v_mbcnt_lo_u32_b32{{(_e[0-9]+)?}} v[[mbcnt:[0-9]+]], s[[exec_lo]], 0
; GCN64: v_mbcnt_hi_u32_b32{{(_e[0-9]+)?}} v[[mbcnt]], s[[exec_hi]], v[[mbcnt]]
; GCN: v_cmp_eq_u32{{(_e[0-9]+)?}} vcc{{(_lo)?}}, 0, v[[mbcnt]]
; GCN: s_and_saveexec_b{{32|64}} s[[exec:\[?[0-9:]+\]?]], vcc
; GCN32: s_bcnt1_i32_b32 s[[popcount:[0-9]+]], s[[exec_lo]]
; GCN64: s_bcnt1_i32_b64 s[[popcount:[0-9]+]], s{{\[}}[[exec_lo]]:[[exec_hi]]{{\]}}
; GCN: v_mul_hi_u32_u24{{(_e[0-9]+)?}} v[[value_hi:[0-9]+]], s[[popcount]], 5
; GCN: v_mul_u32_u24{{(_e[0-9]+)?}} v[[value_lo:[0-9]+]], s[[popcount]], 5
; GCN: {{flat|buffer|global}}_atomic_add_x2 v{{\[}}[[value_lo]]:[[value_hi]]{{\]}}
define amdgpu_kernel void @add_i64_constant(i64 addrspace(1)* %out, i64 addrspace(1)* %inout) {
entry:
  %old = atomicrmw add i64 addrspace(1)* %inout, i64 5 acq_rel
  store i64 %old, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: add_i64_uniform:
; GCN32: s_mov_b32 s[[exec_lo:[0-9]+]], exec_lo
; GCN64: s_mov_b64 s{{\[}}[[exec_lo:[0-9]+]]:[[exec_hi:[0-9]+]]{{\]}}, exec
; GCN: v_mbcnt_lo_u32_b32{{(_e[0-9]+)?}} v[[mbcnt:[0-9]+]], s[[exec_lo]], 0
; GCN64: v_mbcnt_hi_u32_b32{{(_e[0-9]+)?}} v[[mbcnt]], s[[exec_hi]], v[[mbcnt]]
; GCN: v_cmp_eq_u32{{(_e[0-9]+)?}} vcc{{(_lo)?}}, 0, v[[mbcnt]]
; GCN: s_and_saveexec_b{{32|64}} s[[exec:\[?[0-9:]+\]?]], vcc
; GCN32: s_bcnt1_i32_b32 s[[popcount:[0-9]+]], s[[exec_lo]]
; GCN64: s_bcnt1_i32_b64 s[[popcount:[0-9]+]], s{{\[}}[[exec_lo]]:[[exec_hi]]{{\]}}
; GCN: {{flat|buffer|global}}_atomic_add_x2 v{{\[}}{{[0-9]+}}:{{[0-9]+}}{{\]}}
define amdgpu_kernel void @add_i64_uniform(i64 addrspace(1)* %out, i64 addrspace(1)* %inout, i64 %additive) {
entry:
  %old = atomicrmw add i64 addrspace(1)* %inout, i64 %additive acq_rel
  store i64 %old, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: add_i64_varying:
; GCN-NOT: v_mbcnt_lo_u32_b32
; GCN-NOT: v_mbcnt_hi_u32_b32
; GCN-NOT: s_bcnt1_i32_b64
; GCN: {{flat|buffer|global}}_atomic_add_x2 v{{\[}}{{[0-9]+}}:{{[0-9]+}}{{\]}}
define amdgpu_kernel void @add_i64_varying(i64 addrspace(1)* %out, i64 addrspace(1)* %inout) {
entry:
  %lane = call i32 @llvm.amdgcn.workitem.id.x()
  %zext = zext i32 %lane to i64
  %old = atomicrmw add i64 addrspace(1)* %inout, i64 %zext acq_rel
  store i64 %old, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: sub_i32_constant:
; GCN32: s_mov_b32 s[[exec_lo:[0-9]+]], exec_lo
; GCN64: s_mov_b64 s{{\[}}[[exec_lo:[0-9]+]]:[[exec_hi:[0-9]+]]{{\]}}, exec
; GCN: v_mbcnt_lo_u32_b32{{(_e[0-9]+)?}} v[[mbcnt:[0-9]+]], s[[exec_lo]], 0
; GCN64: v_mbcnt_hi_u32_b32{{(_e[0-9]+)?}} v[[mbcnt]], s[[exec_hi]], v[[mbcnt]]
; GCN: v_cmp_eq_u32{{(_e[0-9]+)?}} vcc{{(_lo)?}}, 0, v[[mbcnt]]
; GCN: s_and_saveexec_b{{32|64}} s[[exec:\[?[0-9:]+\]?]], vcc
; GCN32: s_bcnt1_i32_b32 s[[popcount:[0-9]+]], s[[exec_lo]]
; GCN64: s_bcnt1_i32_b64 s[[popcount:[0-9]+]], s{{\[}}[[exec_lo]]:[[exec_hi]]{{\]}}
; GCN: v_mul_u32_u24{{(_e[0-9]+)?}} v[[value:[0-9]+]], s[[popcount]], 5
; GCN: {{flat|buffer|global}}_atomic_sub v[[value]]
define amdgpu_kernel void @sub_i32_constant(i32 addrspace(1)* %out, i32 addrspace(1)* %inout) {
entry:
  %old = atomicrmw sub i32 addrspace(1)* %inout, i32 5 acq_rel
  store i32 %old, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: sub_i32_uniform:
; GCN32: s_mov_b32 s[[exec_lo:[0-9]+]], exec_lo
; GCN64: s_mov_b64 s{{\[}}[[exec_lo:[0-9]+]]:[[exec_hi:[0-9]+]]{{\]}}, exec
; GCN: v_mbcnt_lo_u32_b32{{(_e[0-9]+)?}} v[[mbcnt:[0-9]+]], s[[exec_lo]], 0
; GCN64: v_mbcnt_hi_u32_b32{{(_e[0-9]+)?}} v[[mbcnt]], s[[exec_hi]], v[[mbcnt]]
; GCN: v_cmp_eq_u32{{(_e[0-9]+)?}} vcc{{(_lo)?}}, 0, v[[mbcnt]]
; GCN: s_and_saveexec_b{{32|64}} s[[exec:\[?[0-9:]+\]?]], vcc
; GCN32: s_bcnt1_i32_b32 s[[popcount:[0-9]+]], s[[exec_lo]]
; GCN64: s_bcnt1_i32_b64 s[[popcount:[0-9]+]], s{{\[}}[[exec_lo]]:[[exec_hi]]{{\]}}
; GCN: s_mul_i32 s[[scalar_value:[0-9]+]], s{{[0-9]+}}, s[[popcount]]
; GCN: v_mov_b32{{(_e[0-9]+)?}} v[[value:[0-9]+]], s[[scalar_value]]
; GCN: {{flat|buffer|global}}_atomic_sub v[[value]]
define amdgpu_kernel void @sub_i32_uniform(i32 addrspace(1)* %out, i32 addrspace(1)* %inout, i32 %subitive) {
entry:
  %old = atomicrmw sub i32 addrspace(1)* %inout, i32 %subitive acq_rel
  store i32 %old, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: sub_i32_varying:
; GFX7LESS-NOT: v_mbcnt_lo_u32_b32
; GFX7LESS-NOT: v_mbcnt_hi_u32_b32
; GFX7LESS-NOT: s_bcnt1_i32_b64
; GFX7LESS: buffer_atomic_sub v{{[0-9]+}}
; DPPCOMB: v_add_u32_dpp
; DPPCOMB: v_add_u32_dpp
; GFX8MORE32: v_readlane_b32 s[[scalar_value:[0-9]+]], v{{[0-9]+}}, 31
; GFX8MORE64: v_readlane_b32 s[[scalar_value:[0-9]+]], v{{[0-9]+}}, 63
; GFX89: v_mov_b32{{(_e[0-9]+)?}} v[[value:[0-9]+]], s[[scalar_value]]
; GFX10: s_mov_b32 s[[copy_value:[0-9]+]], s[[scalar_value]]
; GFX10: v_mov_b32{{(_e[0-9]+)?}} v[[value:[0-9]+]], s[[copy_value]]
; GFX8MORE: buffer_atomic_sub v[[value]]
define amdgpu_kernel void @sub_i32_varying(i32 addrspace(1)* %out, i32 addrspace(1)* %inout) {
entry:
  %lane = call i32 @llvm.amdgcn.workitem.id.x()
  %old = atomicrmw sub i32 addrspace(1)* %inout, i32 %lane acq_rel
  store i32 %old, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: sub_i64_constant:
; GCN32: s_mov_b32 s[[exec_lo:[0-9]+]], exec_lo
; GCN64: s_mov_b64 s{{\[}}[[exec_lo:[0-9]+]]:[[exec_hi:[0-9]+]]{{\]}}, exec
; GCN: v_mbcnt_lo_u32_b32{{(_e[0-9]+)?}} v[[mbcnt:[0-9]+]], s[[exec_lo]], 0
; GCN64: v_mbcnt_hi_u32_b32{{(_e[0-9]+)?}} v[[mbcnt]], s[[exec_hi]], v[[mbcnt]]
; GCN: v_cmp_eq_u32{{(_e[0-9]+)?}} vcc{{(_lo)?}}, 0, v[[mbcnt]]
; GCN: s_and_saveexec_b{{32|64}} s[[exec:\[?[0-9:]+\]?]], vcc
; GCN32: s_bcnt1_i32_b32 s[[popcount:[0-9]+]], s[[exec_lo]]
; GCN64: s_bcnt1_i32_b64 s[[popcount:[0-9]+]], s{{\[}}[[exec_lo]]:[[exec_hi]]{{\]}}
; GCN: v_mul_hi_u32_u24{{(_e[0-9]+)?}} v[[value_hi:[0-9]+]], s[[popcount]], 5
; GCN: v_mul_u32_u24{{(_e[0-9]+)?}} v[[value_lo:[0-9]+]], s[[popcount]], 5
; GCN: {{flat|buffer|global}}_atomic_sub_x2 v{{\[}}[[value_lo]]:[[value_hi]]{{\]}}
define amdgpu_kernel void @sub_i64_constant(i64 addrspace(1)* %out, i64 addrspace(1)* %inout) {
entry:
  %old = atomicrmw sub i64 addrspace(1)* %inout, i64 5 acq_rel
  store i64 %old, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: sub_i64_uniform:
; GCN32: s_mov_b32 s[[exec_lo:[0-9]+]], exec_lo
; GCN64: s_mov_b64 s{{\[}}[[exec_lo:[0-9]+]]:[[exec_hi:[0-9]+]]{{\]}}, exec
; GCN: v_mbcnt_lo_u32_b32{{(_e[0-9]+)?}} v[[mbcnt:[0-9]+]], s[[exec_lo]], 0
; GCN64: v_mbcnt_hi_u32_b32{{(_e[0-9]+)?}} v[[mbcnt]], s[[exec_hi]], v[[mbcnt]]
; GCN: v_cmp_eq_u32{{(_e[0-9]+)?}} vcc{{(_lo)?}}, 0, v[[mbcnt]]
; GCN: s_and_saveexec_b{{32|64}} s[[exec:\[?[0-9:]+\]?]], vcc
; GCN32: s_bcnt1_i32_b32 s[[popcount:[0-9]+]], s[[exec_lo]]
; GCN64: s_bcnt1_i32_b64 s[[popcount:[0-9]+]], s{{\[}}[[exec_lo]]:[[exec_hi]]{{\]}}
; GCN: {{flat|buffer|global}}_atomic_sub_x2 v{{\[}}{{[0-9]+}}:{{[0-9]+}}{{\]}}
define amdgpu_kernel void @sub_i64_uniform(i64 addrspace(1)* %out, i64 addrspace(1)* %inout, i64 %subitive) {
entry:
  %old = atomicrmw sub i64 addrspace(1)* %inout, i64 %subitive acq_rel
  store i64 %old, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: sub_i64_varying:
; GCN-NOT: v_mbcnt_lo_u32_b32
; GCN-NOT: v_mbcnt_hi_u32_b32
; GCN-NOT: s_bcnt1_i32_b64
; GCN: {{flat|buffer|global}}_atomic_sub_x2 v{{\[}}{{[0-9]+}}:{{[0-9]+}}{{\]}}
define amdgpu_kernel void @sub_i64_varying(i64 addrspace(1)* %out, i64 addrspace(1)* %inout) {
entry:
  %lane = call i32 @llvm.amdgcn.workitem.id.x()
  %zext = zext i32 %lane to i64
  %old = atomicrmw sub i64 addrspace(1)* %inout, i64 %zext acq_rel
  store i64 %old, i64 addrspace(1)* %out
  ret void
}
