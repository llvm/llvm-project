; RUN: llc -mtriple amdgcn--amdhsa -mcpu=fiji -amdgpu-scalarize-global-loads < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: @volatile_load
; GCN:  s_load_dwordx2 s{{\[}}[[LO_SREG:[0-9]+]]:[[HI_SREG:[0-9]+]]{{\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0x0
; GCN:  v_mov_b32_e32 v[[LO_VREG:[0-9]+]], s[[LO_SREG]]
; GCN:  v_mov_b32_e32 v[[HI_VREG:[0-9]+]], s[[HI_SREG]]
; GCN:  flat_load_dword v{{[0-9]+}}, v{{\[}}[[LO_VREG]]:[[HI_VREG]]{{\]}}

define amdgpu_kernel void @volatile_load(i32 addrspace(1)* %arg, i32 addrspace(1)* nocapture %arg1) {
bb:
  %tmp18 = load volatile i32, i32 addrspace(1)* %arg, align 4
  %tmp26 = getelementptr inbounds i32, i32 addrspace(1)* %arg1, i64 5
  store i32 %tmp18, i32 addrspace(1)* %tmp26, align 4
  ret void
}
