; RUN: llc -global-isel=0 -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SI,NOTGFX9,GCN-SDAG %s
; XUN: llc -global-isel=1 -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SI,NOTGFX9,GCN-GISEL %s
; RUN: llc -global-isel=0 -march=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,CIPLUS,NOTGFX9,CIPLUS-SDAG,GCN-SDAG %s
; RUN: llc -global-isel=1 -march=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,CIPLUS,NOTGFX9,CIPLUS-GISEL,GCN-GISEL %s
; RUN: llc -global-isel=0 -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,CIPLUS,NOTGFX9,CIPLUS-SDAG,GCN-SDAG %s
; RUN: llc -global-isel=1 -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,CIPLUS,NOTGFX9,CIPLUS-GISEL,GCN-GISEL %s
; RUN: llc -global-isel=0 -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,CIPLUS,GFX9,CIPLUS-SDAG,GCN-SDAG %s
; RUN: llc -global-isel=1 -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,CIPLUS,GFX9,CIPLUS-GISEL,GCN-GISEL %s

; GCN-LABEL: {{^}}ds_append_lds:
; GCN: s_load_dword [[PTR:s[0-9]+]]
; GCN: s_mov_b32 m0, [[PTR]]
; GCN: ds_append [[RESULT:v[0-9]+]]{{$}}
; GCN-NOT: buffer_wbinvl1
; GCN: {{.*}}store{{.*}} [[RESULT]]
define amdgpu_kernel void @ds_append_lds(i32 addrspace(3)* %lds, i32 addrspace(1)* %out) #0 {
  %val = call i32 @llvm.amdgcn.ds.append.p3i32(i32 addrspace(3)* %lds, i1 false)
  store i32 %val, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}ds_append_lds_max_offset:
; GCN: s_load_dword [[PTR:s[0-9]+]]
; GCN: s_mov_b32 m0, [[PTR]]
; GCN: ds_append [[RESULT:v[0-9]+]] offset:65532{{$}}
; GCN-NOT: buffer_wbinvl1
; GCN: {{.*}}store{{.*}} [[RESULT]]
define amdgpu_kernel void @ds_append_lds_max_offset(i32 addrspace(3)* %lds, i32 addrspace(1)* %out) #0 {
  %gep = getelementptr inbounds i32, i32 addrspace(3)* %lds, i32 16383
  %val = call i32 @llvm.amdgcn.ds.append.p3i32(i32 addrspace(3)* %gep, i1 false)
  store i32 %val, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}ds_append_no_fold_offset_si:
; GCN: s_load_dword [[PTR:s[0-9]+]]

; SI: s_add_i32 [[PTR]], [[PTR]], 16
; SI: s_mov_b32 m0, [[PTR]]
; SI: ds_append [[RESULT:v[0-9]+]]{{$}}

; CIPLUS: s_mov_b32 m0, [[PTR]]
; CIPLUS: ds_append [[RESULT:v[0-9]+]] offset:16{{$}}

; GCN-NOT: buffer_wbinvl1
; GCN: {{.*}}store{{.*}} [[RESULT]]
define amdgpu_kernel void @ds_append_no_fold_offset_si(i32 addrspace(3)* addrspace(4)* %lds.ptr, i32 addrspace(1)* %out) #0 {
  %lds = load i32 addrspace(3)*, i32 addrspace(3)* addrspace(4)* %lds.ptr, align 4
  %gep = getelementptr inbounds i32, i32 addrspace(3)* %lds, i32 4
  %val = call i32 @llvm.amdgcn.ds.append.p3i32(i32 addrspace(3)* %gep, i1 false)
  store i32 %val, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}ds_append_lds_over_max_offset:
; GCN: s_load_dword [[PTR:s[0-9]+]]

; SI-SDAG: s_bitset1_b32 [[PTR]], 16
; CIPLUS-SDAG: s_add_i32 [[PTR]], [[PTR]], 0x10000
; GCN-SDAG: s_mov_b32 m0, [[PTR]]

; SI-GISEL: s_bitset1_b32 m0, 16
; CIPLUS-GISEL: s_add_u32 m0, [[PTR]], 0x10000

; GCN: ds_append [[RESULT:v[0-9]+]]{{$}}
; GCN-NOT: buffer_wbinvl1
; GCN: {{.*}}store{{.*}} [[RESULT]]
define amdgpu_kernel void @ds_append_lds_over_max_offset(i32 addrspace(3)* %lds, i32 addrspace(1)* %out) #0 {
  %gep = getelementptr inbounds i32, i32 addrspace(3)* %lds, i32 16384
  %val = call i32 @llvm.amdgcn.ds.append.p3i32(i32 addrspace(3)* %gep, i1 false)
  store i32 %val, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}ds_append_lds_vgpr_addr:
; GCN-SDAG: v_readfirstlane_b32 [[READLANE:s[0-9]+]], v0
; GCN-SDAG: s_mov_b32 m0, [[READLANE]]

; GCN-GISEL: v_readfirstlane_b32 m0, v0

; GCN: ds_append [[RESULT:v[0-9]+]]{{$}}
; GCN-NOT: buffer_wbinvl1
; GCN: {{.*}}store{{.*}} [[RESULT]]
define void @ds_append_lds_vgpr_addr(i32 addrspace(3)* %lds, i32 addrspace(1)* %out) #0 {
  %val = call i32 @llvm.amdgcn.ds.append.p3i32(i32 addrspace(3)* %lds, i1 false)
  store i32 %val, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}ds_append_gds:
; GCN: s_load_dword [[PTR:s[0-9]+]]
; GCN: s_mov_b32 m0, [[PTR]]
; GCN: ds_append [[RESULT:v[0-9]+]] gds{{$}}
; GCN-NOT: buffer_wbinvl1
; GCN: {{.*}}store{{.*}} [[RESULT]]
define amdgpu_kernel void @ds_append_gds(i32 addrspace(2)* %gds, i32 addrspace(1)* %out) #0 {
  %val = call i32 @llvm.amdgcn.ds.append.p2i32(i32 addrspace(2)* %gds, i1 false)
  store i32 %val, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}ds_append_gds_max_offset:
; GCN: s_load_dword [[PTR:s[0-9]+]]
; GCN: s_mov_b32 m0, [[PTR]]
; GCN: ds_append [[RESULT:v[0-9]+]] offset:65532 gds{{$}}
; GCN-NOT: buffer_wbinvl1
; GCN: {{.*}}store{{.*}} [[RESULT]]
define amdgpu_kernel void @ds_append_gds_max_offset(i32 addrspace(2)* %gds, i32 addrspace(1)* %out) #0 {
  %gep = getelementptr inbounds i32, i32 addrspace(2)* %gds, i32 16383
  %val = call i32 @llvm.amdgcn.ds.append.p2i32(i32 addrspace(2)* %gep, i1 false)
  store i32 %val, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}ds_append_gds_over_max_offset:
; GCN-NOT: buffer_wbinvl1
define amdgpu_kernel void @ds_append_gds_over_max_offset(i32 addrspace(2)* %gds, i32 addrspace(1)* %out) #0 {
  %gep = getelementptr inbounds i32, i32 addrspace(2)* %gds, i32 16384
  %val = call i32 @llvm.amdgcn.ds.append.p2i32(i32 addrspace(2)* %gep, i1 false)
  store i32 %val, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}ds_append_lds_m0_restore:
; GCN: s_load_dword [[PTR:s[0-9]+]]
; GCN: s_mov_b32 m0, [[PTR]]
; GCN: ds_append [[RESULT:v[0-9]+]]{{$}}
; GCN-NOT: buffer_wbinvl1
; NOTGFX9: s_mov_b32 m0, -1
; GFX9-NOT: m0
; GCN: _store_dword
; GCN: ds_read_b32
define amdgpu_kernel void @ds_append_lds_m0_restore(i32 addrspace(3)* %lds, i32 addrspace(1)* %out) #0 {
  %val0 = call i32 @llvm.amdgcn.ds.append.p3i32(i32 addrspace(3)* %lds, i1 false)
  store i32 %val0, i32 addrspace(1)* %out
  %val1 = load volatile i32, i32 addrspace(3)* %lds
  ret void
}

; Make sure this selects successfully with no use. The result register needs to be constrained.
; GCN-LABEL: {{^}}ds_append_lds_no_use:
; GCN: s_load_dword [[PTR:s[0-9]+]]
; GCN: s_mov_b32 m0, [[PTR]]
; GCN: ds_append [[RESULT:v[0-9]+]] offset:65532{{$}}
define amdgpu_kernel void @ds_append_lds_no_use(i32 addrspace(3)* %lds, i32 addrspace(1)* %out) #0 {
  %gep = getelementptr inbounds i32, i32 addrspace(3)* %lds, i32 16383
  %val = call i32 @llvm.amdgcn.ds.append.p3i32(i32 addrspace(3)* %gep, i1 false)
  ret void
}

declare i32 @llvm.amdgcn.ds.append.p3i32(i32 addrspace(3)* nocapture, i1 immarg) #1
declare i32 @llvm.amdgcn.ds.append.p2i32(i32 addrspace(2)* nocapture, i1 immarg) #1

attributes #0 = { nounwind }
attributes #1 = { argmemonly convergent nounwind }
