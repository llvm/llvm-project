; RUN: llc -mtriple=amdgcn -mcpu=gfx900 < %s | FileCheck %s --check-prefixes=GCN,GFX9
; RUN: llc -mtriple=amdgcn -mcpu=gfx1030 < %s | FileCheck %s --check-prefixes=GCN,GFX10

@lds.0 = internal addrspace(3) global [64 x float] poison, align 16
@lds.1 = internal addrspace(3) global [64 x float] poison, align 16
@lds.2 = internal addrspace(3) global [64 x float] poison, align 16
@lds.3 = internal addrspace(3) global [64 x float] poison, align 16
@lds.4 = internal addrspace(3) global [64 x float] poison, align 16
@lds.5 = internal addrspace(3) global [64 x float] poison, align 16
@lds.6 = internal addrspace(3) global [64 x float] poison, align 16
@lds.7 = internal addrspace(3) global [64 x float] poison, align 16
@lds.8 = internal addrspace(3) global [64 x float] poison, align 16
@lds.9 = internal addrspace(3) global [64 x float] poison, align 16

declare void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> %rsrc, ptr addrspace(3) nocapture, i32 %size, i32 %voffset, i32 %soffset, i32 %offset, i32 %aux)
declare void @llvm.amdgcn.global.load.lds(ptr addrspace(1) nocapture %gptr, ptr addrspace(3) nocapture %lptr, i32 %size, i32 %offset, i32 %aux)

; GCN-LABEL: {{^}}buffer_load_lds_dword_2_arrays:
; GCN-COUNT-4: buffer_load_dword
; GCN: s_waitcnt vmcnt(2)
; GCN: ds_read_b32
; GCN: s_waitcnt vmcnt(0)
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
  %tmp.0 = insertelement <2 x float> poison, float %val.0, i32 0
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
; GFX10: s_waitcnt vmcnt(2)
; GFX10: ds_read_b32
; GFX10: s_waitcnt vmcnt(0)
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
  %tmp.0 = insertelement <2 x float> poison, float %val.0, i32 0
  %res = insertelement <2 x float> %tmp.0, float %val.1, i32 1
  store <2 x float> %res, ptr addrspace(1) %out
  ret void
}

; There are 8 pseudo registers defined to track LDS DMA dependencies.

; GCN-LABEL: {{^}}buffer_load_lds_dword_10_arrays:
; GCN-COUNT-10: buffer_load_dword
; GCN: s_waitcnt vmcnt(8)
; GCN: ds_read_b32
; GCN: s_waitcnt vmcnt(7)
; GCN: ds_read_b32
; GCN: s_waitcnt vmcnt(6)
; GCN: ds_read_b32
; GCN: s_waitcnt vmcnt(5)
; GCN: ds_read_b32
; GCN: s_waitcnt vmcnt(4)
; GCN: ds_read_b32
; GCN: s_waitcnt vmcnt(3)
; GCN: ds_read_b32
; GCN: s_waitcnt vmcnt(2)
; GCN-NOT: s_waitcnt vmcnt
; GCN: ds_read_b32
; GCN: ds_read_b32
define amdgpu_kernel void @buffer_load_lds_dword_10_arrays(<4 x i32> %rsrc, i32 %i1, i32 %i2, i32 %i3, i32 %i4, i32 %i5, i32 %i6, i32 %i7, i32 %i8, i32 %i9, ptr addrspace(1) %out) {
main_body:
  call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> %rsrc, ptr addrspace(3) @lds.0, i32 4, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> %rsrc, ptr addrspace(3) @lds.1, i32 4, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> %rsrc, ptr addrspace(3) @lds.2, i32 4, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> %rsrc, ptr addrspace(3) @lds.3, i32 4, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> %rsrc, ptr addrspace(3) @lds.4, i32 4, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> %rsrc, ptr addrspace(3) @lds.5, i32 4, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> %rsrc, ptr addrspace(3) @lds.6, i32 4, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> %rsrc, ptr addrspace(3) @lds.7, i32 4, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> %rsrc, ptr addrspace(3) @lds.8, i32 4, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> %rsrc, ptr addrspace(3) @lds.9, i32 4, i32 0, i32 0, i32 0, i32 0)
  %gep.0 = getelementptr float, ptr addrspace(3) @lds.0, i32 %i1
  %gep.1 = getelementptr float, ptr addrspace(3) @lds.1, i32 %i2
  %gep.2 = getelementptr float, ptr addrspace(3) @lds.2, i32 %i2
  %gep.3 = getelementptr float, ptr addrspace(3) @lds.3, i32 %i2
  %gep.4 = getelementptr float, ptr addrspace(3) @lds.4, i32 %i2
  %gep.5 = getelementptr float, ptr addrspace(3) @lds.5, i32 %i2
  %gep.6 = getelementptr float, ptr addrspace(3) @lds.6, i32 %i2
  %gep.7 = getelementptr float, ptr addrspace(3) @lds.7, i32 %i2
  %gep.8 = getelementptr float, ptr addrspace(3) @lds.8, i32 %i2
  %gep.9 = getelementptr float, ptr addrspace(3) @lds.9, i32 %i2
  %val.0 = load float, ptr addrspace(3) %gep.0, align 4
  call void @llvm.amdgcn.wave.barrier()
  %val.1 = load float, ptr addrspace(3) %gep.1, align 4
  call void @llvm.amdgcn.wave.barrier()
  %val.2 = load float, ptr addrspace(3) %gep.2, align 4
  call void @llvm.amdgcn.wave.barrier()
  %val.3 = load float, ptr addrspace(3) %gep.3, align 4
  call void @llvm.amdgcn.wave.barrier()
  %val.4 = load float, ptr addrspace(3) %gep.4, align 4
  call void @llvm.amdgcn.wave.barrier()
  %val.5 = load float, ptr addrspace(3) %gep.5, align 4
  call void @llvm.amdgcn.wave.barrier()
  %val.6 = load float, ptr addrspace(3) %gep.6, align 4
  call void @llvm.amdgcn.wave.barrier()
  %val.7 = load float, ptr addrspace(3) %gep.7, align 4
  call void @llvm.amdgcn.wave.barrier()
  %val.8 = load float, ptr addrspace(3) %gep.8, align 4
  call void @llvm.amdgcn.wave.barrier()
  %val.9 = load float, ptr addrspace(3) %gep.9, align 4
  %out.gep.1 = getelementptr float, ptr addrspace(1) %out, i32 1
  %out.gep.2 = getelementptr float, ptr addrspace(1) %out, i32 2
  %out.gep.3 = getelementptr float, ptr addrspace(1) %out, i32 3
  %out.gep.4 = getelementptr float, ptr addrspace(1) %out, i32 4
  %out.gep.5 = getelementptr float, ptr addrspace(1) %out, i32 5
  %out.gep.6 = getelementptr float, ptr addrspace(1) %out, i32 6
  %out.gep.7 = getelementptr float, ptr addrspace(1) %out, i32 7
  %out.gep.8 = getelementptr float, ptr addrspace(1) %out, i32 8
  %out.gep.9 = getelementptr float, ptr addrspace(1) %out, i32 9
  store float %val.0, ptr addrspace(1) %out
  store float %val.1, ptr addrspace(1) %out.gep.1
  store float %val.2, ptr addrspace(1) %out.gep.2
  store float %val.3, ptr addrspace(1) %out.gep.3
  store float %val.4, ptr addrspace(1) %out.gep.4
  store float %val.5, ptr addrspace(1) %out.gep.5
  store float %val.6, ptr addrspace(1) %out.gep.6
  store float %val.7, ptr addrspace(1) %out.gep.7
  store float %val.8, ptr addrspace(1) %out.gep.8
  store float %val.9, ptr addrspace(1) %out.gep.9
  ret void
}

define amdgpu_kernel void @global_load_lds_no_alias_ds_read(ptr addrspace(1) nocapture %gptr, i32 %i1, i32 %i2, ptr addrspace(1) %out) {
; GFX9-LABEL: global_load_lds_no_alias_ds_read:
; GFX9: global_load_dword
; GFX9: global_load_dword
; GFX9: s_waitcnt vmcnt(1)
; GFX9-NOT: s_waitcnt vmcnt(0)
; GFX9: ds_read_b32
; GFX9: s_waitcnt vmcnt(0)
; GFX9: ds_read_b32
; GFX9: s_endpgm
body:
  call void @llvm.amdgcn.global.load.lds(ptr addrspace(1) %gptr, ptr addrspace(3) @lds.0, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.global.load.lds(ptr addrspace(1) %gptr, ptr addrspace(3) @lds.1, i32 4, i32 4, i32 0)
  call void @llvm.amdgcn.s.waitcnt(i32 3953)
  %gep.0 = getelementptr float, ptr addrspace(3) @lds.2, i32 %i1
  %val.0 = load float, ptr addrspace(3) %gep.0, align 4
  call void @llvm.amdgcn.s.waitcnt(i32 3952)
  %gep.1 = getelementptr float, ptr addrspace(3) @lds.3, i32 %i2
  %val.1 = load float, ptr addrspace(3) %gep.1, align 4
  %tmp = insertelement <2 x float> poison, float %val.0, i32 0
  %res = insertelement <2 x float> %tmp, float %val.1, i32 1
  store <2 x float> %res, ptr addrspace(1) %out
  ret void
}

declare void @llvm.amdgcn.wave.barrier()
