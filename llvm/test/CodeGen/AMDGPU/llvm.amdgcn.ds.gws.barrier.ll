; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=tahiti -o - -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,LOOP %s
; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=hawaii -o - -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,LOOP %s
; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=fiji -o - -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,LOOP %s
; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=gfx900 -o - -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,NOLOOP %s
; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=gfx1010 -o - -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,NOLOOP,GFX10 %s

; Minimum offset
; GCN-LABEL: {{^}}gws_barrier_offset0:
; NOLOOP-DAG: s_load_dword [[BAR_NUM:s[0-9]+]]
; NOLOOP-DAG: s_mov_b32 m0, -1{{$}}
; NOLOOP: v_mov_b32_e32 v0, [[BAR_NUM]]
; NOLOOP: ds_gws_barrier v0 offset:1 gds{{$}}

; LOOP: [[LOOP:BB[0-9]+_[0-9]+]]:
; LOOP-NEXT: s_setreg_imm32_b32 hwreg(HW_REG_TRAPSTS, 8, 1), 0
; LOOP-NEXT: ds_gws_barrier v0 offset:1 gds
; LOOP-NEXT: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; LOOP-NEXT: s_getreg_b32 [[GETREG:s[0-9]+]], hwreg(HW_REG_TRAPSTS, 8, 1)
; LOOP-NEXT: s_cmp_lg_u32 [[GETREG]], 0
; LOOP-NEXT: s_cbranch_scc1 [[LOOP]]
define amdgpu_kernel void @gws_barrier_offset0(i32 %val) #0 {
  call void @llvm.amdgcn.ds.gws.barrier(i32 %val, i32 0)
  ret void
}

; Maximum offset
; GCN-LABEL: {{^}}gws_barrier_offset63:
; NOLOOP-DAG: s_load_dword [[BAR_NUM:s[0-9]+]]
; NOLOOP-DAG: s_mov_b32 m0, -1{{$}}
; NOLOOP-DAG: v_mov_b32_e32 v0, [[BAR_NUM]]
; NOLOOP: ds_gws_barrier v0 offset:64 gds{{$}}
define amdgpu_kernel void @gws_barrier_offset63(i32 %val) #0 {
  call void @llvm.amdgcn.ds.gws.barrier(i32 %val, i32 63)
  ret void
}

; FIXME: Should be able to shift directly into m0
; GCN-LABEL: {{^}}gws_barrier_sgpr_offset:
; NOLOOP-DAG: s_load_dwordx2 s{{\[}}[[BAR_NUM:[0-9]+]]:[[OFFSET:[0-9]+]]{{\]}}
; NOLOOP-DAG: s_lshl_b32 [[SHL:s[0-9]+]], s[[OFFSET]], 16
; NOLOOP-DAG: s_mov_b32 m0, [[SHL]]{{$}}
; NOLOOP-DAG: v_mov_b32_e32 v0, s[[BAR_NUM]]
; NOLOOP: ds_gws_barrier v0 gds{{$}}
define amdgpu_kernel void @gws_barrier_sgpr_offset(i32 %val, i32 %offset) #0 {
  call void @llvm.amdgcn.ds.gws.barrier(i32 %val, i32 %offset)
  ret void
}

; Variable offset in SGPR with constant add
; GCN-LABEL: {{^}}gws_barrier_sgpr_offset_add1:
; NOLOOP-DAG: s_load_dwordx2 s{{\[}}[[BAR_NUM:[0-9]+]]:[[OFFSET:[0-9]+]]{{\]}}
; NOLOOP-DAG: s_lshl_b32 [[SHL:s[0-9]+]], s[[OFFSET]], 16
; NOLOOP-DAG: s_mov_b32 m0, [[SHL]]{{$}}
; NOLOOP-DAG: v_mov_b32_e32 v0, s[[BAR_NUM]]
; NOLOOP: ds_gws_barrier v0 offset:1 gds{{$}}
define amdgpu_kernel void @gws_barrier_sgpr_offset_add1(i32 %val, i32 %offset.base) #0 {
  %offset = add i32 %offset.base, 1
  call void @llvm.amdgcn.ds.gws.barrier(i32 %val, i32 %offset)
  ret void
}

; GCN-LABEL: {{^}}gws_barrier_vgpr_offset:
; NOLOOP-DAG: s_load_dword [[BAR_NUM:s[0-9]+]]
; NOLOOP-DAG: v_readfirstlane_b32 [[READLANE:s[0-9]+]], v0
; NOLOOP-DAG: s_lshl_b32 [[SHL:s[0-9]+]], [[READLANE]], 16
; NOLOOP-DAG: s_mov_b32 m0, [[SHL]]{{$}}
; NOLOOP-DAG: v_mov_b32_e32 v0, [[BAR_NUM]]
; NOLOOP: ds_gws_barrier v0 gds{{$}}
define amdgpu_kernel void @gws_barrier_vgpr_offset(i32 %val) #0 {
  %vgpr.offset = call i32 @llvm.amdgcn.workitem.id.x()
  call void @llvm.amdgcn.ds.gws.barrier(i32 %val, i32 %vgpr.offset)
  ret void
}

; Variable offset in VGPR with constant add
; GCN-LABEL: {{^}}gws_barrier_vgpr_offset_add:
; NOLOOP-DAG: s_load_dword [[BAR_NUM:s[0-9]+]]
; NOLOOP-DAG: v_readfirstlane_b32 [[READLANE:s[0-9]+]], v0
; NOLOOP-DAG: s_lshl_b32 [[SHL:s[0-9]+]], [[READLANE]], 16
; NOLOOP-DAG: s_mov_b32 m0, [[SHL]]{{$}}
; NOLOOP-DAG: v_mov_b32_e32 v0, [[BAR_NUM]]
; NOLOOP: ds_gws_barrier v0 offset:3 gds{{$}}
define amdgpu_kernel void @gws_barrier_vgpr_offset_add(i32 %val) #0 {
  %vgpr.offset.base = call i32 @llvm.amdgcn.workitem.id.x()
  %vgpr.offset = add i32 %vgpr.offset.base, 3
  call void @llvm.amdgcn.ds.gws.barrier(i32 %val, i32 %vgpr.offset)
  ret void
}

@lds = internal unnamed_addr addrspace(3) global i32 undef

; Check if m0 initialization is shared
; GCN-LABEL: {{^}}gws_barrier_save_m0_barrier_constant_offset:
; NOLOOP: s_mov_b32 m0, -1
; NOLOOP-NOT: s_mov_b32 m0
define amdgpu_kernel void @gws_barrier_save_m0_barrier_constant_offset(i32 %val) #0 {
  store i32 1, i32 addrspace(3)* @lds
  call void @llvm.amdgcn.ds.gws.barrier(i32 %val, i32 10)
  store i32 2, i32 addrspace(3)* @lds
  ret void
}

; Make sure this increments lgkmcnt
; GCN-LABEL: {{^}}gws_barrier_lgkmcnt:
; NOLOOP: ds_gws_barrier v0 offset:1 gds{{$}}
; NOLOOP-NEXT: s_waitcnt expcnt(0) lgkmcnt(0)
; NOLOOP-NEXT: s_setpc_b64
define void @gws_barrier_lgkmcnt(i32 %val) {
  call void @llvm.amdgcn.ds.gws.barrier(i32 %val, i32 0)
  ret void
}

; Does not imply memory fence on its own
; GCN-LABEL: {{^}}gws_barrier_wait_before:
; NOLOOP: s_waitcnt
; NOLOOP-NOT: s_waitcnt{{$}}
define amdgpu_kernel void @gws_barrier_wait_before(i32 %val, i32 addrspace(1)* %ptr) #0 {
  store i32 0, i32 addrspace(1)* %ptr
  call void @llvm.amdgcn.ds.gws.barrier(i32 %val, i32 7)
  ret void
}

; GCN-LABEL: {{^}}gws_barrier_wait_after:
; NOLOOP: ds_gws_barrier v0 offset:8 gds
; NOLOOP-NEXT: s_waitcnt expcnt(0){{$}}
; NOLOOP-NEXT: load_dword
define amdgpu_kernel void @gws_barrier_wait_after(i32 %val, i32 addrspace(1)* %ptr) #0 {
  call void @llvm.amdgcn.ds.gws.barrier(i32 %val, i32 7)
  %load = load volatile i32, i32 addrspace(1)* %ptr
  ret void
}

; Does not imply memory fence on its own
; GCN-LABEL: {{^}}gws_barrier_fence_before:
; NOLOOP: store_dword
; NOLOOP: s_waitcnt vmcnt(0) lgkmcnt(0)
; NOLOOP: ds_gws_barrier v0 offset:8 gds
define amdgpu_kernel void @gws_barrier_fence_before(i32 %val, i32 addrspace(1)* %ptr) #0 {
  store i32 0, i32 addrspace(1)* %ptr
  fence release
  call void @llvm.amdgcn.ds.gws.barrier(i32 %val, i32 7)
  ret void
}

; GCN-LABEL: {{^}}gws_barrier_fence_after:
; NOLOOP: ds_gws_barrier v0 offset:8 gds
; NOLOOP-NEXT: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0
; NOLOOP-NEXT: load_dword
define amdgpu_kernel void @gws_barrier_fence_after(i32 %val, i32 addrspace(1)* %ptr) #0 {
  call void @llvm.amdgcn.ds.gws.barrier(i32 %val, i32 7)
  fence release
  %load = load volatile i32, i32 addrspace(1)* %ptr
  ret void
}

; FIXME: Should a wait be inserted here, or is an explicit fence needed?
; GCN-LABEL: {{^}}gws_init_barrier:
; NOLOOP: s_mov_b32 m0, -1
; NOLOOP: ds_gws_init v0 offset:8 gds
; NOLOOP-NEXT: ds_gws_barrier v0 offset:8 gds
define amdgpu_kernel void @gws_init_barrier(i32 %val) #0 {
  call void @llvm.amdgcn.ds.gws.init(i32 %val, i32 7)
  call void @llvm.amdgcn.ds.gws.barrier(i32 %val, i32 7)
  ret void
}

; FIXME: Why vmcnt, not expcnt?
; GCN-LABEL: {{^}}gws_init_fence_barrier:
; NOLOOP: s_mov_b32 m0, -1
; NOLOOP: ds_gws_init v0 offset:8 gds
; NOLOOP-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0)
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0
; NOLOOP-NEXT: ds_gws_barrier v0 offset:8 gds
define amdgpu_kernel void @gws_init_fence_barrier(i32 %val) #0 {
  call void @llvm.amdgcn.ds.gws.init(i32 %val, i32 7)
  fence release
  call void @llvm.amdgcn.ds.gws.barrier(i32 %val, i32 7)
  ret void
}

declare void @llvm.amdgcn.ds.gws.barrier(i32, i32) #1
declare void @llvm.amdgcn.ds.gws.init(i32, i32) #2
declare i32 @llvm.amdgcn.workitem.id.x() #3

attributes #0 = { nounwind }
attributes #1 = { convergent inaccessiblememonly nounwind }
attributes #2 = { convergent inaccessiblememonly nounwind writeonly }
attributes #3 = { nounwind readnone speculatable }
