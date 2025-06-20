; RUN: llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx90a < %s | FileCheck --check-prefixes=GCN,GFX90A %s
; RUN: llc -global-isel=1 -global-isel-abort=2 -mtriple=amdgcn -mcpu=gfx90a < %s | FileCheck --check-prefixes=GCN,GFX90A %s
; RUN: llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx942 < %s | FileCheck --check-prefixes=GCN,GFX942 %s
; RUN: llc -global-isel=1 -global-isel-abort=2 -mtriple=amdgcn -mcpu=gfx942 < %s | FileCheck --check-prefixes=GCN,GFX942 %s
; RUN: llc -global-isel=1 -global-isel-abort=2 -mtriple=amdgcn -mcpu=gfx9-4-generic --amdhsa-code-object-version=6 < %s | FileCheck --check-prefixes=GCN,GFX942 %s

; DPP control value 337 is valid for 64-bit DPP on gfx942

; GCN-LABEL: update_dpp_i64:
;
; GFX90A-DAG:    v_mov_b32_dpp v2, v0 row_newbcast:1 row_mask:0x1 bank_mask:0x1
; GFX90A-DAG:    v_mov_b32_dpp v3, v1 row_newbcast:1 row_mask:0x1 bank_mask:0x1
;
; GFX942:        v_mov_b64_dpp v[2:3], v[0:1] row_newbcast:1 row_mask:0x1 bank_mask:0x1
;
; GCN:           global_store_dwordx2 v[4:5], v[2:3], off

define amdgpu_ps void @update_dpp_i64(i64 %in, i64 %old, ptr addrspace(1) %out) {
  %tmp0 = call i64 @llvm.amdgcn.update.dpp.i64(i64 %old, i64 %in, i32 337, i32 1, i32 1, i1 0)
  store i64 %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: update_dpp_v2i32:
;
; GFX90A-DAG:    v_mov_b32_dpp v2, v0 row_newbcast:1 row_mask:0x1 bank_mask:0x1
; GFX90A-DAG:    v_mov_b32_dpp v3, v1 row_newbcast:1 row_mask:0x1 bank_mask:0x1
;
; GFX942:        v_mov_b64_dpp v[2:3], v[0:1] row_newbcast:1 row_mask:0x1 bank_mask:0x1
;
; GCN:           global_store_dwordx2 v[4:5], v[2:3], off

define amdgpu_ps void @update_dpp_v2i32(<2 x i32> %in, <2 x i32> %old, ptr addrspace(1) %out) {
  %tmp0 = call <2 x i32> @llvm.amdgcn.update.dpp.v2i32(<2 x i32> %old, <2 x i32> %in, i32 337, i32 1, i32 1, i1 0)
  store <2 x i32> %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: update_dpp_v3i32:
;
; GCN-DAG:    v_mov_b32_dpp v{{[0-9]+}}, v2 row_newbcast:1 row_mask:0x1 bank_mask:0x1
; GCN-DAG:    v_mov_b32_dpp v{{[0-9]+}}, v1 row_newbcast:1 row_mask:0x1 bank_mask:0x1
; GCN-DAG:    v_mov_b32_dpp v{{[0-9]+}}, v0 row_newbcast:1 row_mask:0x1 bank_mask:0x1
;
; GCN:        global_store_dwordx3

define amdgpu_ps void @update_dpp_v3i32(<3 x i32> %in, <3 x i32> %old, ptr addrspace(1) %out) {
  %tmp0 = call <3 x i32> @llvm.amdgcn.update.dpp.v3i32(<3 x i32> %old, <3 x i32> %in, i32 337, i32 1, i32 1, i1 0)
  store <3 x i32> %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: update_dpp_v4i32:
;
; GFX90A-DAG:    v_mov_b32_dpp v6, v2 row_newbcast:1 row_mask:0x1 bank_mask:0x1
; GFX90A-DAG:    v_mov_b32_dpp v7, v3 row_newbcast:1 row_mask:0x1 bank_mask:0x1
; GFX90A-DAG:    v_mov_b32_dpp v4, v0 row_newbcast:1 row_mask:0x1 bank_mask:0x1
; GFX90A-DAG:    v_mov_b32_dpp v5, v1 row_newbcast:1 row_mask:0x1 bank_mask:0x1
;
; GFX942-DAG:    v_mov_b64_dpp v[6:7], v[2:3] row_newbcast:1 row_mask:0x1 bank_mask:0x1
; GFX942-DAG:    v_mov_b64_dpp v[4:5], v[0:1] row_newbcast:1 row_mask:0x1 bank_mask:0x1
;
; GCN:           global_store_dwordx4 v[8:9], v[4:7], off

define amdgpu_ps void @update_dpp_v4i32(<4 x i32> %in, <4 x i32> %old, ptr addrspace(1) %out) {
  %tmp0 = call <4 x i32> @llvm.amdgcn.update.dpp.v4i32(<4 x i32> %old, <4 x i32> %in, i32 337, i32 1, i32 1, i1 0)
  store <4 x i32> %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: update_dpp_v2i32_poison:
;
; GFX90A-DAG:    v_mov_b32_dpp v0, v0 row_newbcast:1 row_mask:0x1 bank_mask:0x1
; GFX90A-DAG:    v_mov_b32_dpp v1, v1 row_newbcast:1 row_mask:0x1 bank_mask:0x1
;
; GFX942:        v_mov_b64_dpp v[0:1], v[0:1] row_newbcast:1 row_mask:0x1 bank_mask:0x1
;
; GCN:           global_store_dwordx2 v[2:3], v[0:1], off

define amdgpu_ps void @update_dpp_v2i32_poison(<2 x i32> %in, ptr addrspace(1) %out) {
  %tmp0 = call <2 x i32> @llvm.amdgcn.update.dpp.v2i32(<2 x i32> poison, <2 x i32> %in, i32 337, i32 1, i32 1, i1 0)
  store <2 x i32> %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: update_dpp_float:
;
; GCN:        v_mov_b32_dpp v1, v0 row_newbcast:1 row_mask:0x1 bank_mask:0x1
;
; GCN:        global_store_dword v[2:3], v1, off

define amdgpu_ps void @update_dpp_float(float %in, float %old, ptr addrspace(1) %out) {
  %tmp0 = call float @llvm.amdgcn.update.dpp.f32(float %old, float %in, i32 337, i32 1, i32 1, i1 0)
  store float %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: update_dpp_double:
;
; GFX90A-DAG:    v_mov_b32_dpp v2, v0 row_newbcast:1 row_mask:0x1 bank_mask:0x1
; GFX90A-DAG:    v_mov_b32_dpp v3, v1 row_newbcast:1 row_mask:0x1 bank_mask:0x1
;
; GFX942:        v_mov_b64_dpp v[2:3], v[0:1] row_newbcast:1 row_mask:0x1 bank_mask:0x1
;
; GCN:           global_store_dwordx2 v[4:5], v[2:3], off

define amdgpu_ps void @update_dpp_double(double %in, double %old, ptr addrspace(1) %out) {
  %tmp0 = call double @llvm.amdgcn.update.dpp.f64(double %old, double %in, i32 337, i32 1, i32 1, i1 0)
  store double %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: update_dpp_float_poison:
;
; GCN:        v_mov_b32_dpp v0, v0 row_newbcast:1 row_mask:0x1 bank_mask:0x1
;
; GCN:        global_store_dword v[{{[0-9:]+}}], v0, off

define amdgpu_ps void @update_dpp_float_poison(float %in, ptr addrspace(1) %out) {
  %tmp0 = call float @llvm.amdgcn.update.dpp.f32(float poison, float %in, i32 337, i32 1, i32 1, i1 0)
  store float %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: update_dpp_double_poison:
;
; GFX90A-DAG:    v_mov_b32_dpp v0, v0 row_newbcast:1 row_mask:0x1 bank_mask:0x1
; GFX90A-DAG:    v_mov_b32_dpp v1, v1 row_newbcast:1 row_mask:0x1 bank_mask:0x1
;
; GFX942:        v_mov_b64_dpp v[0:1], v[0:1] row_newbcast:1 row_mask:0x1 bank_mask:0x1
;
; GCN:           global_store_dwordx2 v[2:3], v[0:1], off

define amdgpu_ps void @update_dpp_double_poison(double %in, ptr addrspace(1) %out) {
  %tmp0 = call double @llvm.amdgcn.update.dpp.f64(double poison, double %in, i32 337, i32 1, i32 1, i1 0)
  store double %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: update_dpp_v2f32_poison:
;
; GFX90A-DAG:    v_mov_b32_dpp v0, v0 row_newbcast:1 row_mask:0x1 bank_mask:0x1
; GFX90A-DAG:    v_mov_b32_dpp v1, v1 row_newbcast:1 row_mask:0x1 bank_mask:0x1
;
; GFX942:        v_mov_b64_dpp v[0:1], v[0:1] row_newbcast:1 row_mask:0x1 bank_mask:0x1
;
; GCN:           global_store_dwordx2 v[2:3], v[0:1], off

define amdgpu_ps void @update_dpp_v2f32_poison(<2 x float> %in, ptr addrspace(1) %out) {
  %tmp0 = call <2 x float> @llvm.amdgcn.update.dpp.v2f32(<2 x float> poison, <2 x float> %in, i32 337, i32 1, i32 1, i1 0)
  store <2 x float> %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: update_dpp_v2f16_poison:
;
; GCN:         v_mov_b32_dpp v0, v0 row_newbcast:1 row_mask:0x1 bank_mask:0x1
;
; GCN:         global_store_dword v[{{[0-9:]+}}], v0, off

define amdgpu_ps void @update_dpp_v2f16_poison(<2 x half> %in, ptr addrspace(1) %out) {
  %tmp0 = call <2 x half> @llvm.amdgcn.update.dpp.v2f16(<2 x half> poison, <2 x half> %in, i32 337, i32 1, i32 1, i1 0)
  store <2 x half> %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: update_dpp_v8f16_poison:
;
; GFX90A-DAG:    v_mov_b32_dpp v0, v0 row_newbcast:1 row_mask:0x1 bank_mask:0x1
; GFX90A-DAG:    v_mov_b32_dpp v1, v1 row_newbcast:1 row_mask:0x1 bank_mask:0x1
; GFX90A-DAG:    v_mov_b32_dpp v2, v2 row_newbcast:1 row_mask:0x1 bank_mask:0x1
; GFX90A-DAG:    v_mov_b32_dpp v3, v3 row_newbcast:1 row_mask:0x1 bank_mask:0x1
;
; GFX942-DAG:    v_mov_b64_dpp v[2:3], v[2:3] row_newbcast:1 row_mask:0x1 bank_mask:0x1
; GFX942-DAG:    v_mov_b64_dpp v[0:1], v[0:1] row_newbcast:1 row_mask:0x1 bank_mask:0x1
;
; GCN:           global_store_dwordx4 v[4:5], v[0:3], off

define amdgpu_ps void @update_dpp_v8f16_poison(<8 x half> %in, ptr addrspace(1) %out) {
  %tmp0 = call <8 x half> @llvm.amdgcn.update.dpp.v8f16(<8 x half> poison, <8 x half> %in, i32 337, i32 1, i32 1, i1 0)
  store <8 x half> %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: update_dpp_p3_poison:
;
; GCN:        v_mov_b32_dpp v0, v0 row_newbcast:1 row_mask:0x1 bank_mask:0x1
;
; GCN:        global_store_dword v[{{[0-9:]+}}], v0, off

define amdgpu_ps void @update_dpp_p3_poison(ptr addrspace(3) %in, ptr addrspace(1) %out) {
  %tmp0 = call ptr addrspace(3) @llvm.amdgcn.update.dpp.p3(ptr addrspace(3) poison, ptr addrspace(3) %in, i32 337, i32 1, i32 1, i1 0)
  store ptr addrspace(3) %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: update_dpp_p0_poison:
;
; GFX90A-DAG:    v_mov_b32_dpp v0, v0 row_newbcast:1 row_mask:0x1 bank_mask:0x1
; GFX90A-DAG:    v_mov_b32_dpp v1, v1 row_newbcast:1 row_mask:0x1 bank_mask:0x1
;
; GFX942-DAG:    v_mov_b64_dpp v[0:1], v[0:1] row_newbcast:1 row_mask:0x1 bank_mask:0x1
;
; GCN:            global_store_dwordx2 v[2:3], v[0:1], off

define amdgpu_ps void @update_dpp_p0_poison(ptr %in, ptr addrspace(1) %out) {
  %tmp0 = call ptr @llvm.amdgcn.update.dpp.p0(ptr poison, ptr %in, i32 337, i32 1, i32 1, i1 0)
  store ptr %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: update_dpp_i64_unsupported_dpp64_op:
;
; GCN-DAG:    v_mov_b32_dpp v3, v1 quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1
; GCN-DAG:    v_mov_b32_dpp v2, v0 quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1
;
; GCN:        global_store_dwordx2 v[4:5], v[2:3], off

define amdgpu_ps void @update_dpp_i64_unsupported_dpp64_op(i64 %in, i64 %old, ptr addrspace(1) %out) {
  %tmp0 = call i64 @llvm.amdgcn.update.dpp.i64(i64 %old, i64 %in, i32 1, i32 1, i32 1, i1 0)
  store i64 %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: update_dpp_i16:
;
; GCN:        v_mov_b32_dpp v1, v0 row_newbcast:1 row_mask:0x1 bank_mask:0x1
;
; GCN:        global_store_short v[2:3], v1, off

define amdgpu_ps void @update_dpp_i16(i16 %in, i16 %old, ptr addrspace(1) %out) {
  %tmp0 = call i16 @llvm.amdgcn.update.dpp.i16(i16 %old, i16 %in, i32 337, i32 1, i32 1, i1 0)
  store i16 %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: update_dpp_half:
;
; GCN-DAG:    v_mov_b32_dpp v1, v0 row_newbcast:1 row_mask:0x1 bank_mask:0x1
;
; GCN:        global_store_short v[2:3], v1, off

define amdgpu_ps void @update_dpp_half(half %in, half %old, ptr addrspace(1) %out) {
  %tmp0 = call half @llvm.amdgcn.update.dpp.f16(half %old, half %in, i32 337, i32 1, i32 1, i1 0)
  store half %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: update_dpp_bfloat:
;
; GCN-DAG:    v_mov_b32_dpp v1, v0 row_newbcast:1 row_mask:0x1 bank_mask:0x1
;
; GCN:        global_store_short v[2:3], v1, off

define amdgpu_ps void @update_dpp_bfloat(bfloat %in, bfloat %old, ptr addrspace(1) %out) {
  %tmp0 = call bfloat @llvm.amdgcn.update.dpp.bf16(bfloat %old, bfloat %in, i32 337, i32 1, i32 1, i1 0)
  store bfloat %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: update_dpp_v2p0_poison:
;
; GFX90A-DAG:    v_mov_b32_dpp v0, v0 row_newbcast:1 row_mask:0x1 bank_mask:0x2
; GFX90A-DAG:    v_mov_b32_dpp v1, v1 row_newbcast:1 row_mask:0x1 bank_mask:0x2
; GFX90A-DAG:    v_mov_b32_dpp v2, v2 row_newbcast:1 row_mask:0x1 bank_mask:0x2
; GFX90A-DAG:    v_mov_b32_dpp v3, v3 row_newbcast:1 row_mask:0x1 bank_mask:0x2
;
; GFX942-DAG:    v_mov_b64_dpp v[2:3], v[2:3] row_newbcast:1 row_mask:0x1 bank_mask:0x2
; GFX942-DAG:    v_mov_b64_dpp v[0:1], v[0:1] row_newbcast:1 row_mask:0x1 bank_mask:0x2
;
; GCN:           global_store_dwordx4 v[4:5], v[0:3], off

define amdgpu_ps void @update_dpp_v2p0_poison(<2 x ptr> %in, ptr addrspace(1) %out) {
  %tmp0 = call <2 x ptr> @llvm.amdgcn.update.dpp.v2p0(<2 x ptr> poison, <2 x ptr> %in, i32 337, i32 1, i32 2, i1 0)
  store <2 x ptr> %tmp0, ptr addrspace(1) %out
  ret void
}
