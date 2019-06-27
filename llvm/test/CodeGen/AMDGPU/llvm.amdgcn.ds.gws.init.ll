; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=tahiti -o - -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN %s
; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=hawaii -o - -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN %s
; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=fiji -o - -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN %s
; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=gfx900 -o - -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN %s

; Minimum offset
; GCN-LABEL: {{^}}gws_init_offset0:
; GCN-DAG: s_load_dword [[BAR_NUM:s[0-9]+]]
; GCN-DAG: s_mov_b32 m0, -1{{$}}
; GCN: v_mov_b32_e32 v0, [[BAR_NUM]]
; GCN: ds_gws_init v0 offset:1 gds{{$}}
define amdgpu_kernel void @gws_init_offset0(i32 %val) #0 {
  call void @llvm.amdgcn.ds.gws.init(i32 %val, i32 0)
  ret void
}

; Maximum offset
; GCN-LABEL: {{^}}gws_init_offset63:
; GCN-DAG: s_load_dword [[BAR_NUM:s[0-9]+]]
; GCN-DAG: s_mov_b32 m0, -1{{$}}
; GCN-DAG: v_mov_b32_e32 v0, [[BAR_NUM]]
; GCN: ds_gws_init v0 offset:64 gds{{$}}
define amdgpu_kernel void @gws_init_offset63(i32 %val) #0 {
  call void @llvm.amdgcn.ds.gws.init(i32 %val, i32 63)
  ret void
}

; FIXME: Should be able to shift directly into m0
; GCN-LABEL: {{^}}gws_init_sgpr_offset:
; GCN-DAG: s_load_dwordx2 s{{\[}}[[BAR_NUM:[0-9]+]]:[[OFFSET:[0-9]+]]{{\]}}
; GCN-DAG: s_lshl_b32 [[SHL:s[0-9]+]], s[[OFFSET]], 16
; GCN-DAG: s_mov_b32 m0, [[SHL]]{{$}}
; GCN-DAG: v_mov_b32_e32 v0, s[[BAR_NUM]]
; GCN: ds_gws_init v0 gds{{$}}
define amdgpu_kernel void @gws_init_sgpr_offset(i32 %val, i32 %offset) #0 {
  call void @llvm.amdgcn.ds.gws.init(i32 %val, i32 %offset)
  ret void
}

; Variable offset in SGPR with constant add
; GCN-LABEL: {{^}}gws_init_sgpr_offset_add1:
; GCN-DAG: s_load_dwordx2 s{{\[}}[[BAR_NUM:[0-9]+]]:[[OFFSET:[0-9]+]]{{\]}}
; GCN-DAG: s_lshl_b32 [[SHL:s[0-9]+]], s[[OFFSET]], 16
; GCN-DAG: s_mov_b32 m0, [[SHL]]{{$}}
; GCN-DAG: v_mov_b32_e32 v0, s[[BAR_NUM]]
; GCN: ds_gws_init v0 offset:1 gds{{$}}
define amdgpu_kernel void @gws_init_sgpr_offset_add1(i32 %val, i32 %offset.base) #0 {
  %offset = add i32 %offset.base, 1
  call void @llvm.amdgcn.ds.gws.init(i32 %val, i32 %offset)
  ret void
}

; GCN-LABEL: {{^}}gws_init_vgpr_offset:
; GCN-DAG: s_load_dword [[BAR_NUM:s[0-9]+]]
; GCN-DAG: v_readfirstlane_b32 [[READLANE:s[0-9]+]], v0
; GCN-DAG: s_lshl_b32 [[SHL:s[0-9]+]], [[READLANE]], 16
; GCN-DAG: s_mov_b32 m0, [[SHL]]{{$}}
; GCN-DAG: v_mov_b32_e32 v0, [[BAR_NUM]]
; GCN: ds_gws_init v0 gds{{$}}
define amdgpu_kernel void @gws_init_vgpr_offset(i32 %val) #0 {
  %vgpr.offset = call i32 @llvm.amdgcn.workitem.id.x()
  call void @llvm.amdgcn.ds.gws.init(i32 %val, i32 %vgpr.offset)
  ret void
}

; Variable offset in VGPR with constant add
; GCN-LABEL: {{^}}gws_init_vgpr_offset_add:
; GCN-DAG: s_load_dword [[BAR_NUM:s[0-9]+]]
; GCN-DAG: v_readfirstlane_b32 [[READLANE:s[0-9]+]], v0
; GCN-DAG: s_lshl_b32 [[SHL:s[0-9]+]], [[READLANE]], 16
; GCN-DAG: s_mov_b32 m0, [[SHL]]{{$}}
; GCN-DAG: v_mov_b32_e32 v0, [[BAR_NUM]]
; GCN: ds_gws_init v0 offset:3 gds{{$}}
define amdgpu_kernel void @gws_init_vgpr_offset_add(i32 %val) #0 {
  %vgpr.offset.base = call i32 @llvm.amdgcn.workitem.id.x()
  %vgpr.offset = add i32 %vgpr.offset.base, 3
  call void @llvm.amdgcn.ds.gws.init(i32 %val, i32 %vgpr.offset)
  ret void
}

@lds = internal unnamed_addr addrspace(3) global i32 undef

; Check if m0 initialization is shared.
; GCN-LABEL: {{^}}gws_init_save_m0_init_constant_offset:
; GCN: s_mov_b32 m0, -1
; GCN-NOT: s_mov_b32 m0
define amdgpu_kernel void @gws_init_save_m0_init_constant_offset(i32 %val) #0 {
  store i32 1, i32 addrspace(3)* @lds
  call void @llvm.amdgcn.ds.gws.init(i32 %val, i32 10)
  store i32 2, i32 addrspace(3)* @lds
  ret void
}

; GCN-LABEL: {{^}}gws_init_lgkmcnt:
; GCN: ds_gws_init v0 offset:1 gds{{$}}
; GCN-NEXT: s_waitcnt expcnt(0) lgkmcnt(0)
; GCN-NEXT: s_setpc_b64
define void @gws_init_lgkmcnt(i32 %val) {
  call void @llvm.amdgcn.ds.gws.init(i32 %val, i32 0)
  ret void
}

; Does not imply memory fence on its own
; GCN-LABEL: {{^}}gws_init_wait_before:
; GCN: store_dword
; CIPLUS-NOT: s_waitcnt
; GCN: ds_gws_init v0 offset:8 gds
define amdgpu_kernel void @gws_init_wait_before(i32 %val, i32 addrspace(1)* %ptr) #0 {
  store i32 0, i32 addrspace(1)* %ptr
  call void @llvm.amdgcn.ds.gws.init(i32 %val, i32 7)
  ret void
}

declare void @llvm.amdgcn.ds.gws.init(i32, i32) #1
declare i32 @llvm.amdgcn.workitem.id.x() #2

attributes #0 = { nounwind }
attributes #1 = { convergent inaccessiblememonly nounwind writeonly }
attributes #2 = { nounwind readnone speculatable }
