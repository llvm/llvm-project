; RUN: llc -march=amdgcn -mcpu=gfx900 < %s | FileCheck %s --check-prefixes=GCN,GFX9
; RUN: llc -march=amdgcn -mcpu=gfx1030 < %s | FileCheck %s --check-prefixes=GCN,GFX10

@lds.0 = internal addrspace(3) global [64 x float] poison, align 16
@lds.1 = internal addrspace(3) global [64 x float] poison, align 16

declare void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> %rsrc, ptr addrspace(3) nocapture, i32 %size, i32 %voffset, i32 %soffset, i32 %offset, i32 %aux)
declare void @llvm.amdgcn.global.load.lds(ptr addrspace(1) nocapture %gptr, ptr addrspace(3) nocapture %lptr, i32 %size, i32 %offset, i32 %aux)

; FIXME: vmcnt(0) is too strong, it shall use vmcnt(2) before the first
;        ds_read_b32 and vmcnt(0) before the second.

; GCN-LABEL: {{^}}buffer_load_lds_dword_2_arrays:
; GCN-COUNT-4: buffer_load_dword
; GCN: s_waitcnt vmcnt(0)
; GCN: ds_read_b32

; FIXME:
; GCN-NOT: s_waitcnt

; GCN: ds_read_b32
define amdgpu_kernel void @buffer_load_lds_dword_2_arrays(<4 x i32> %rsrc, i32 %i1, i32 %i2, ptr addrspace(1) %out) {
main_body:
  call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> %rsrc, ptr addrspace(3) @lds.0, i32 4, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> %rsrc, ptr addrspace(3) @lds.0, i32 4, i32 4, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> %rsrc, ptr addrspace(3) @lds.1, i32 4, i32 8, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> %rsrc, ptr addrspace(3) @lds.1, i32 4, i32 12, i32 0, i32 0, i32 0)
  %gep.0 = getelementptr float, ptr addrspace(3) @lds.0, i32 %i1
  %gep.1 = getelementptr float, ptr addrspace(3) @lds.1, i32 %i2
  %val.0 = load float, ptr addrspace(3) %gep.0, align 4
  call void @llvm.amdgcn.wave.barrier()
  %val.1 = load float, ptr addrspace(3) %gep.1, align 4
  %tmp.0 = insertelement <2 x float> undef, float %val.0, i32 0
  %res = insertelement <2 x float> %tmp.0, float %val.1, i32 1
  store <2 x float> %res, ptr addrspace(1) %out
  ret void
}

; On gfx9 if there is a pending FLAT operation, and this is a VMem or LGKM
; waitcnt and the target can report early completion, then we need to force a waitcnt 0.

; GCN-LABEL: {{^}}global_load_lds_dword_2_arrays:
; GCN-COUNT-4: global_load_dword
; GFX9: s_waitcnt vmcnt(0)
; GFX9-COUNT-2: ds_read_b32

; FIXME: can be vmcnt(2)

; GFX10: s_waitcnt vmcnt(0)
; GFX10: ds_read_b32

; FIXME:
; GFX10-NOT: s_waitcnt

; GFX10: ds_read_b32
define amdgpu_kernel void @global_load_lds_dword_2_arrays(ptr addrspace(1) nocapture %gptr, i32 %i1, i32 %i2, ptr addrspace(1) %out) {
main_body:
  call void @llvm.amdgcn.global.load.lds(ptr addrspace(1) %gptr, ptr addrspace(3) @lds.0, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.global.load.lds(ptr addrspace(1) %gptr, ptr addrspace(3) @lds.0, i32 4, i32 4, i32 0)
  call void @llvm.amdgcn.global.load.lds(ptr addrspace(1) %gptr, ptr addrspace(3) @lds.1, i32 4, i32 8, i32 0)
  call void @llvm.amdgcn.global.load.lds(ptr addrspace(1) %gptr, ptr addrspace(3) @lds.1, i32 4, i32 12, i32 0)
  %gep.0 = getelementptr float, ptr addrspace(3) @lds.0, i32 %i1
  %gep.1 = getelementptr float, ptr addrspace(3) @lds.1, i32 %i2
  %val.0 = load float, ptr addrspace(3) %gep.0, align 4
  call void @llvm.amdgcn.wave.barrier()
  %val.1 = load float, ptr addrspace(3) %gep.1, align 4
  %tmp.0 = insertelement <2 x float> undef, float %val.0, i32 0
  %res = insertelement <2 x float> %tmp.0, float %val.1, i32 1
  store <2 x float> %res, ptr addrspace(1) %out
  ret void
}

declare void @llvm.amdgcn.wave.barrier()
