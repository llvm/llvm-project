; RUN: llc -march=amdgcn -mcpu=kaveri -mtriple=amdgcn-unknown-amdhsa -mattr=-flat-for-global < %s | FileCheck -check-prefix=GCN %s

; Check that when mubuf addr64 instruction is handled in moveToVALU
; from the pointer, dead register writes are not emitted.

; FIXME: We should be able to use the SGPR directly as src0 to v_add_i32

; GCN-LABEL: {{^}}clobber_vgpr_pair_pointer_add:
; GCN-DAG: buffer_load_dwordx2 v{{\[}}[[LDPTRLO:[0-9]+]]:[[LDPTRHI:[0-9]+]]{{\]}}
; GCN-DAG: s_load_dwordx2 s{{\[}}[[ARG1LO:[0-9]+]]:[[ARG1HI:[0-9]+]]{{\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0x0{{$}}

; GCN-DAG: v_mov_b32_e32 v[[VARG1LO:[0-9]+]], s[[ARG1LO]]
; GCN-DAG: v_mov_b32_e32 v[[VARG1HI:[0-9]+]], s[[ARG1HI]]
; GCN-NOT: v_mov_b32
; GCN-NOT: v_mov_b32

; GCN: v_add_i32_e32 v[[PTRLO:[0-9]+]], vcc, v[[LDPTRLO]], v[[VARG1LO]]
; GCN: v_addc_u32_e32 v[[PTRHI:[0-9]+]], vcc, v[[LDPTRHI]], v[[VARG1HI]]
; GCN: buffer_load_ubyte v{{[0-9]+}}, v{{\[}}[[PTRLO]]:[[PTRHI]]{{\]}},

define amdgpu_kernel void @clobber_vgpr_pair_pointer_add(i64 %arg1, [8 x i32], i8 addrspace(1)* addrspace(1)* %ptrarg, i32 %arg3) #0 {
bb:
  %tmp = icmp sgt i32 %arg3, 0
  br i1 %tmp, label %bb4, label %bb17

bb4:
  %tmp14 = load volatile i8 addrspace(1)*, i8 addrspace(1)* addrspace(1)* %ptrarg
  %tmp15 = getelementptr inbounds i8, i8 addrspace(1)* %tmp14, i64 %arg1
  %tmp16 = load volatile i8, i8 addrspace(1)* %tmp15
  br label %bb17

bb17:
  ret void
}

attributes #0 = { nounwind }
