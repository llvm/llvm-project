; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=gfx908 -o - -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,GFX908 %s
; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=gfx90a -o - -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,GFX90A %s
; RUN: llc -global-isel -mtriple=amdgcn-mesa-mesa3d -mcpu=gfx908 -o - -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,GFX908 %s
; RUN: llc -global-isel -mtriple=amdgcn-mesa-mesa3d -mcpu=gfx90a -o - -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,GFX90A %s
; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=gfx908 -early-live-intervals -o - -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,GFX908 %s

; GCN-LABEL: {{^}}gws_init_odd_reg:
; GFX908-DAG: ds_gws_init v1 gds
; GFX90A-DAG: ds_gws_init v2 gds
; GCN-DAG:    ds_gws_init v0 gds
define amdgpu_ps void @gws_init_odd_reg(<2 x i32> %arg) {
  %vgpr.0 = extractelement <2 x i32> %arg, i32 0
  %vgpr.1 = extractelement <2 x i32> %arg, i32 1
  call void @llvm.amdgcn.ds.gws.init(i32 %vgpr.0, i32 0)
  call void @llvm.amdgcn.ds.gws.init(i32 %vgpr.1, i32 0)
  ret void
}

; GCN-LABEL: {{^}}gws_sema_br_odd_reg:
; GFX908-DAG: ds_gws_sema_br v1 gds
; GFX90A-DAG: ds_gws_sema_br v2 gds
; GCN-DAG:    ds_gws_sema_br v0 gds
define amdgpu_ps void @gws_sema_br_odd_reg(<2 x i32> %arg) {
  %vgpr.0 = extractelement <2 x i32> %arg, i32 0
  %vgpr.1 = extractelement <2 x i32> %arg, i32 1
  call void @llvm.amdgcn.ds.gws.sema.br(i32 %vgpr.0, i32 0)
  call void @llvm.amdgcn.ds.gws.sema.br(i32 %vgpr.1, i32 0)
  ret void
}

; GCN-LABEL: {{^}}gws_barrier_odd_reg:
; GFX908-DAG: ds_gws_barrier v1 gds
; GFX90A-DAG: ds_gws_barrier v2 gds
; GCN-DAG:    ds_gws_barrier v0 gds
define amdgpu_ps void @gws_barrier_odd_reg(<2 x i32> %arg) {
  %vgpr.0 = extractelement <2 x i32> %arg, i32 0
  %vgpr.1 = extractelement <2 x i32> %arg, i32 1
  call void @llvm.amdgcn.ds.gws.barrier(i32 %vgpr.0, i32 0)
  call void @llvm.amdgcn.ds.gws.barrier(i32 %vgpr.1, i32 0)
  ret void
}

; GCN-LABEL: {{^}}gws_init_odd_agpr:
; GFX908-COUNT-2: ds_gws_init v{{[0-9]+}} gds
; GFX90A-COUNT-2: ds_gws_init {{[va][0-9]?[02468]}} gds
define amdgpu_ps void @gws_init_odd_agpr(<4 x i32> %arg) {
bb:
  %mai = tail call <4 x i32> @llvm.amdgcn.mfma.i32.4x4x4i8(i32 1, i32 2, <4 x i32> %arg, i32 0, i32 0, i32 0)
  %agpr.0 = extractelement <4 x i32> %mai, i32 0
  %agpr.1 = extractelement <4 x i32> %mai, i32 1
  call void @llvm.amdgcn.ds.gws.init(i32 %agpr.0, i32 0)
  call void @llvm.amdgcn.ds.gws.init(i32 %agpr.1, i32 0)
  ret void
}

declare void @llvm.amdgcn.ds.gws.init(i32, i32)
declare void @llvm.amdgcn.ds.gws.sema.br(i32, i32)
declare void @llvm.amdgcn.ds.gws.barrier(i32, i32)
declare <4 x i32> @llvm.amdgcn.mfma.i32.4x4x4i8(i32, i32, <4 x i32>, i32, i32, i32)
