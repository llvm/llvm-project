; RUN: llc -global-isel=0 -verify-machineinstrs -mtriple=amdgcn -mcpu=gfx9900 -mattr=+wavefrontsize32 < %s | FileCheck %s

; --- i32, GVS addressing (kernel args -> SGPR base) ---
define amdgpu_kernel void @pred_atomic_load_i32_gvs(ptr addrspace(1) %ptr, i1 %pred, ptr addrspace(1) %out) {
; CHECK-LABEL: pred_atomic_load_i32_gvs:
; CHECK: s_and_saveexec_b32
; CHECK: global_load_b32 v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9:]+}}]
; CHECK: s_mov_b32 exec_lo,
  %val = call i32 @llvm.amdgcn.predicated.atomic.load.i32(ptr addrspace(1) %ptr, i1 %pred)
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; --- v2i32, GVS addressing ---
define amdgpu_kernel void @pred_atomic_load_v2i32_gvs(ptr addrspace(1) %ptr, i1 %pred, ptr addrspace(1) %out) {
; CHECK-LABEL: pred_atomic_load_v2i32_gvs:
; CHECK: s_and_saveexec_b32
; CHECK: global_load_b64 v[{{[0-9:]+}}], v{{[0-9]+}}, s[{{[0-9:]+}}]
; CHECK: s_mov_b32 exec_lo,
  %val = call <2 x i32> @llvm.amdgcn.predicated.atomic.load.v2i32(ptr addrspace(1) %ptr, i1 %pred)
  store <2 x i32> %val, ptr addrspace(1) %out
  ret void
}

; --- v4i32, GVS addressing ---
define amdgpu_kernel void @pred_atomic_load_v4i32_gvs(ptr addrspace(1) %ptr, i1 %pred, ptr addrspace(1) %out) {
; CHECK-LABEL: pred_atomic_load_v4i32_gvs:
; CHECK: s_and_saveexec_b32
; CHECK: global_load_b128 v[{{[0-9:]+}}], v{{[0-9]+}}, s[{{[0-9:]+}}]
; CHECK: s_mov_b32 exec_lo,
  %val = call <4 x i32> @llvm.amdgcn.predicated.atomic.load.v4i32(ptr addrspace(1) %ptr, i1 %pred)
  store <4 x i32> %val, ptr addrspace(1) %out
  ret void
}

; --- i32, GV addressing (divergent pointer via workitem.id) ---
define void @pred_atomic_load_i32_gv(ptr addrspace(1) %ptr, i1 %pred, ptr addrspace(1) %out) {
; CHECK-LABEL: pred_atomic_load_i32_gv:
; CHECK: s_and_saveexec_b32
; CHECK: global_load_b32 v{{[0-9]+}}, v[{{[0-9:]+}}], off
; CHECK: s_mov_b32 exec_lo,
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr i32, ptr addrspace(1) %ptr, i32 %tid
  %val = call i32 @llvm.amdgcn.predicated.atomic.load.i32(ptr addrspace(1) %gep, i1 %pred)
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; --- v2i32, GV addressing ---
define void @pred_atomic_load_v2i32_gv(ptr addrspace(1) %ptr, i1 %pred, ptr addrspace(1) %out) {
; CHECK-LABEL: pred_atomic_load_v2i32_gv:
; CHECK: s_and_saveexec_b32
; CHECK: global_load_b64 v[{{[0-9:]+}}], v[{{[0-9:]+}}], off
; CHECK: s_mov_b32 exec_lo,
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr <2 x i32>, ptr addrspace(1) %ptr, i32 %tid
  %val = call <2 x i32> @llvm.amdgcn.predicated.atomic.load.v2i32(ptr addrspace(1) %gep, i1 %pred)
  store <2 x i32> %val, ptr addrspace(1) %out
  ret void
}

; --- v4i32, GV addressing ---
define void @pred_atomic_load_v4i32_gv(ptr addrspace(1) %ptr, i1 %pred, ptr addrspace(1) %out) {
; CHECK-LABEL: pred_atomic_load_v4i32_gv:
; CHECK: s_and_saveexec_b32
; CHECK: global_load_b128 v[{{[0-9:]+}}], v[{{[0-9:]+}}], off
; CHECK: s_mov_b32 exec_lo,
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr <4 x i32>, ptr addrspace(1) %ptr, i32 %tid
  %val = call <4 x i32> @llvm.amdgcn.predicated.atomic.load.v4i32(ptr addrspace(1) %gep, i1 %pred)
  store <4 x i32> %val, ptr addrspace(1) %out
  ret void
}

declare i32 @llvm.amdgcn.predicated.atomic.load.i32(ptr addrspace(1), i1)
declare <2 x i32> @llvm.amdgcn.predicated.atomic.load.v2i32(ptr addrspace(1), i1)
declare <4 x i32> @llvm.amdgcn.predicated.atomic.load.v4i32(ptr addrspace(1), i1)
declare i32 @llvm.amdgcn.workitem.id.x()
