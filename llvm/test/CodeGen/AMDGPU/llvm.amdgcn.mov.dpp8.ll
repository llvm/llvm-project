; RUN: llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefix=GFX10PLUS %s
; RUN: llc -global-isel=1 -global-isel-abort=2 -mtriple=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefix=GFX10PLUS %s
; RUN: llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx1100 -amdgpu-enable-vopd=0 -verify-machineinstrs < %s | FileCheck -check-prefix=GFX10PLUS %s
; RUN: llc -global-isel=1 -global-isel-abort=2 -mtriple=amdgcn -mcpu=gfx1100 -amdgpu-enable-vopd=0 -verify-machineinstrs < %s | FileCheck -check-prefix=GFX10PLUS %s
; RUN: llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx1200 -amdgpu-enable-vopd=0 -verify-machineinstrs < %s | FileCheck -check-prefix=GFX10PLUS %s
; RUN: llc -global-isel=1 -global-isel-abort=2 -mtriple=amdgcn -mcpu=gfx1200 -amdgpu-enable-vopd=0 -verify-machineinstrs < %s | FileCheck -check-prefix=GFX10PLUS %s

; GFX10PLUS-LABEL: {{^}}dpp8_test:
; GFX10PLUS: v_mov_b32_e32 [[SRC:v[0-9]+]], s{{[0-9]+}}
; GFX10PLUS: v_mov_b32_dpp [[SRC]], [[SRC]]  dpp8:[1,0,0,0,0,0,0,0]{{$}}
define amdgpu_kernel void @dpp8_test(ptr addrspace(1) %out, i32 %in) {
  %tmp0 = call i32 @llvm.amdgcn.mov.dpp8.i32(i32 %in, i32 1) #0
  store i32 %tmp0, ptr addrspace(1) %out
  ret void
}

; GFX10PLUS-LABEL: {{^}}dpp8_wait_states:
; GFX10PLUS-NOOPT: v_mov_b32_e32 [[VGPR1:v[0-9]+]], s{{[0-9]+}}
; GFX10PLUS: v_mov_b32_e32 [[VGPR0:v[0-9]+]], s{{[0-9]+}}
; GFX10PLUS: v_mov_b32_dpp [[VGPR0]], [[VGPR0]] dpp8:[1,0,0,0,0,0,0,0]{{$}}
; GFX10PLUS: v_mov_b32_dpp [[VGPR0]], [[VGPR0]] dpp8:[5,0,0,0,0,0,0,0]{{$}}
define amdgpu_kernel void @dpp8_wait_states(ptr addrspace(1) %out, i32 %in) {
  %tmp0 = call i32 @llvm.amdgcn.mov.dpp8.i32(i32 %in, i32 1) #0
  %tmp1 = call i32 @llvm.amdgcn.mov.dpp8.i32(i32 %tmp0, i32 5) #0
  store i32 %tmp1, ptr addrspace(1) %out
  ret void
}

; GFX10PLUS-LABEL: {{^}}dpp8_i64:
; GFX10PLUS-DAG: v_mov_b32_dpp v0, v0 dpp8:[1,0,0,0,0,0,0,0]
; GFX10PLUS-DAG: v_mov_b32_dpp v1, v1 dpp8:[1,0,0,0,0,0,0,0]
; GFX10PLUS-DAG: global_store_{{dwordx2|b64}} v[2:3], v[0:1], off
define amdgpu_ps void @dpp8_i64(i64 %in, ptr addrspace(1) %out) {
  %tmp0 = call i64 @llvm.amdgcn.mov.dpp8.i64(i64 %in, i32 1)
  store i64 %tmp0, ptr addrspace(1) %out
  ret void
}

; GFX10PLUS-LABEL: {{^}}dpp8_v2i32:
; GFX10PLUS-DAG: v_mov_b32_dpp v0, v0 dpp8:[1,0,0,0,0,0,0,0]
; GFX10PLUS-DAG: v_mov_b32_dpp v1, v1 dpp8:[1,0,0,0,0,0,0,0]
; GFX10PLUS-DAG: global_store_{{dwordx2|b64}} v[2:3], v[0:1], off
define amdgpu_ps void @dpp8_v2i32(<2 x i32> %in, ptr addrspace(1) %out) {
  %tmp0 = call <2 x i32> @llvm.amdgcn.mov.dpp8.v3i32(<2 x i32> %in, i32 1)
  store <2 x i32> %tmp0, ptr addrspace(1) %out
  ret void
}

; GFX10PLUS-LABEL: {{^}}dpp8_v3i32:
; GFX10PLUS-DAG: v_mov_b32_dpp v0, v0 dpp8:[1,0,0,0,0,0,0,0]
; GFX10PLUS-DAG: v_mov_b32_dpp v1, v1 dpp8:[1,0,0,0,0,0,0,0]
; GFX10PLUS-DAG: v_mov_b32_dpp v2, v2 dpp8:[1,0,0,0,0,0,0,0]
; GFX10PLUS-DAG: global_store_{{dwordx3|b96}} v[3:4], v[0:2], off
define amdgpu_ps void @dpp8_v3i32(<3 x i32> %in, ptr addrspace(1) %out) {
  %tmp0 = call <3 x i32> @llvm.amdgcn.mov.dpp8.v3i32(<3 x i32> %in, i32 1)
  store <3 x i32> %tmp0, ptr addrspace(1) %out
  ret void
}

; GFX10PLUS-LABEL: {{^}}dpp8_v4i32:
; GFX10PLUS-DAG: v_mov_b32_dpp v0, v0 dpp8:[1,0,0,0,0,0,0,0]
; GFX10PLUS-DAG: v_mov_b32_dpp v1, v1 dpp8:[1,0,0,0,0,0,0,0]
; GFX10PLUS-DAG: v_mov_b32_dpp v2, v2 dpp8:[1,0,0,0,0,0,0,0]
; GFX10PLUS-DAG: v_mov_b32_dpp v3, v3 dpp8:[1,0,0,0,0,0,0,0]
; GFX10PLUS-DAG: global_store_{{dwordx4|b128}} v[4:5], v[0:3], off
define amdgpu_ps void @dpp8_v4i32(<4 x i32> %in, ptr addrspace(1) %out) {
  %tmp0 = call <4 x i32> @llvm.amdgcn.mov.dpp8.v3i32(<4 x i32> %in, i32 1)
  store <4 x i32> %tmp0, ptr addrspace(1) %out
  ret void
}

; GFX10PLUS-LABEL: {{^}}dpp8_p0:
; GFX10PLUS-DAG: v_mov_b32_dpp v0, v0 dpp8:[1,0,0,0,0,0,0,0]
; GFX10PLUS-DAG: v_mov_b32_dpp v1, v1 dpp8:[1,0,0,0,0,0,0,0]
; GFX10PLUS-DAG: global_store_{{dwordx2|b64}} v[2:3], v[0:1], off
define amdgpu_ps void @dpp8_p0(ptr %in, ptr addrspace(1) %out) {
  %tmp0 = call ptr @llvm.amdgcn.mov.dpp8.p0(ptr %in, i32 1)
  store ptr %tmp0, ptr addrspace(1) %out
  ret void
}

; GFX10PLUS-LABEL: {{^}}dpp8_p3:
; GFX10PLUS-DAG: v_mov_b32_dpp v0, v0 dpp8:[1,0,0,0,0,0,0,0]
; GFX10PLUS-DAG: global_store_{{dword|b32}} v[1:2], v0, off
define amdgpu_ps void @dpp8_p3(ptr addrspace(3) %in, ptr addrspace(1) %out) {
  %tmp0 = call ptr addrspace(3) @llvm.amdgcn.mov.dpp8.v3p3(ptr addrspace(3) %in, i32 1)
  store ptr addrspace(3) %tmp0, ptr addrspace(1) %out
  ret void
}

; GFX10PLUS-LABEL: {{^}}dpp8_v3p3:
; GFX10PLUS-DAG: v_mov_b32_dpp v0, v0 dpp8:[1,0,0,0,0,0,0,0]
; GFX10PLUS-DAG: v_mov_b32_dpp v1, v1 dpp8:[1,0,0,0,0,0,0,0]
; GFX10PLUS-DAG: v_mov_b32_dpp v2, v2 dpp8:[1,0,0,0,0,0,0,0]
; GFX10PLUS-DAG: global_store_{{dwordx3|b96}} v[3:4], v[0:2], off
define amdgpu_ps void @dpp8_v3p3(<3 x ptr addrspace(3)> %in, ptr addrspace(1) %out) {
  %tmp0 = call <3 x ptr addrspace(3)> @llvm.amdgcn.mov.dpp8.v3p3(<3 x ptr addrspace(3)> %in, i32 1)
  store <3 x ptr addrspace(3)> %tmp0, ptr addrspace(1) %out
  ret void
}

; GFX10PLUS-LABEL: {{^}}dpp8_i16:
; GFX10PLUS-DAG: v_mov_b32_dpp v0, v0 dpp8:[1,0,0,0,0,0,0,0]
; GFX10PLUS-DAG: global_store_{{short|b16}} v[1:2], v0, off
define amdgpu_ps void @dpp8_i16(i16 %in, ptr addrspace(1) %out) {
  %tmp0 = call i16 @llvm.amdgcn.mov.dpp8.i16(i16 %in, i32 1)
  store i16 %tmp0, ptr addrspace(1) %out
  ret void
}

; GFX10PLUS-LABEL: {{^}}dpp8_v4i16:
; GFX10PLUS-DAG: v_mov_b32_dpp v0, v0 dpp8:[1,0,0,0,0,0,0,0]
; GFX10PLUS-DAG: v_mov_b32_dpp v1, v1 dpp8:[1,0,0,0,0,0,0,0]
; GFX10PLUS-DAG: global_store_{{dwordx2|b64}} v[2:3], v[0:1], off
define amdgpu_ps void @dpp8_v4i16(<4 x i16> %in, ptr addrspace(1) %out) {
  %tmp0 = call <4 x i16> @llvm.amdgcn.mov.dpp8.v4i16(<4 x i16> %in, i32 1)
  store <4 x i16> %tmp0, ptr addrspace(1) %out
  ret void
}

; GFX10PLUS-LABEL: {{^}}dpp8_v4f16:
; GFX10PLUS-DAG: v_mov_b32_dpp v0, v0 dpp8:[1,0,0,0,0,0,0,0]
; GFX10PLUS-DAG: v_mov_b32_dpp v1, v1 dpp8:[1,0,0,0,0,0,0,0]
; GFX10PLUS-DAG: global_store_{{dwordx2|b64}} v[2:3], v[0:1], off
define amdgpu_ps void @dpp8_v4f16(<4 x half> %in, ptr addrspace(1) %out) {
  %tmp0 = call <4 x half> @llvm.amdgcn.mov.dpp8.v4f16(<4 x half> %in, i32 1)
  store <4 x half> %tmp0, ptr addrspace(1) %out
  ret void
}

; GFX10PLUS-LABEL: {{^}}dpp8_float:
; GFX10PLUS-DAG: v_mov_b32_dpp v0, v0 dpp8:[1,0,0,0,0,0,0,0]
; GFX10PLUS-DAG: global_store_{{dword|b32}} v[1:2], v0, off
define amdgpu_ps void @dpp8_float(float %in, ptr addrspace(1) %out) {
  %tmp0 = call float @llvm.amdgcn.mov.dpp8.f32(float %in, i32 1)
  store float %tmp0, ptr addrspace(1) %out
  ret void
}

; GFX10PLUS-LABEL: {{^}}dpp8_v3f32:
; GFX10PLUS-DAG: v_mov_b32_dpp v0, v0 dpp8:[1,0,0,0,0,0,0,0]
; GFX10PLUS-DAG: v_mov_b32_dpp v1, v1 dpp8:[1,0,0,0,0,0,0,0]
; GFX10PLUS-DAG: v_mov_b32_dpp v2, v2 dpp8:[1,0,0,0,0,0,0,0]
; GFX10PLUS-DAG: global_store_{{dwordx3|b96}} v[3:4], v[0:2], off
define amdgpu_ps void @dpp8_v3f32(<3 x float> %in, ptr addrspace(1) %out) {
  %tmp0 = call <3 x float> @llvm.amdgcn.mov.dpp8.v3f32(<3 x float> %in, i32 1)
  store <3 x float> %tmp0, ptr addrspace(1) %out
  ret void
}

; GFX10PLUS-LABEL: {{^}}dpp8_half:
; GFX10PLUS-DAG: v_mov_b32_dpp v0, v0 dpp8:[1,0,0,0,0,0,0,0]
; GFX10PLUS-DAG: global_store_{{short|b16}} v[1:2], v0, off
define amdgpu_ps void @dpp8_half(half %in, ptr addrspace(1) %out) {
  %tmp0 = call half @llvm.amdgcn.mov.dpp8.f16(half %in, i32 1)
  store half %tmp0, ptr addrspace(1) %out
  ret void
}

; GFX10PLUS-LABEL: {{^}}dpp8_bfloat:
; GFX10PLUS-DAG: v_mov_b32_dpp v0, v0 dpp8:[1,0,0,0,0,0,0,0]
; GFX10PLUS-DAG: global_store_{{short|b16}} v[1:2], v0, off
define amdgpu_ps void @dpp8_bfloat(bfloat %in, ptr addrspace(1) %out) {
  %tmp0 = call bfloat @llvm.amdgcn.mov.dpp8.bf16(bfloat %in, i32 1)
  store bfloat %tmp0, ptr addrspace(1) %out
  ret void
}

; GFX10PLUS-LABEL: {{^}}dpp8_v4bf16:
; GFX10PLUS-DAG: v_mov_b32_dpp v0, v0 dpp8:[1,0,0,0,0,0,0,0]
; GFX10PLUS-DAG: v_mov_b32_dpp v1, v1 dpp8:[1,0,0,0,0,0,0,0]
; GFX10PLUS-DAG: global_store_{{dwordx2|b64}} v[2:3], v[0:1], off
define amdgpu_ps void @dpp8_v4bf16(<4 x bfloat> %in, ptr addrspace(1) %out) {
  %tmp0 = call <4 x bfloat> @llvm.amdgcn.mov.dpp8.v4bf16(<4 x bfloat> %in, i32 1)
  store <4 x bfloat> %tmp0, ptr addrspace(1) %out
  ret void
}

; GFX10PLUS-LABEL: {{^}}dpp8_double:
; GFX10PLUS-DAG: v_mov_b32_dpp v0, v0 dpp8:[1,0,0,0,0,0,0,0]
; GFX10PLUS-DAG: v_mov_b32_dpp v1, v1 dpp8:[1,0,0,0,0,0,0,0]
; GFX10PLUS-DAG: global_store_{{dwordx2|b64}} v[2:3], v[0:1], off
define amdgpu_ps void @dpp8_double(double %in, ptr addrspace(1) %out) {
  %tmp0 = call double @llvm.amdgcn.mov.dpp8.f64(double %in, i32 1)
  store double %tmp0, ptr addrspace(1) %out
  ret void
}

declare i32 @llvm.amdgcn.mov.dpp8.i32(i32, i32) #0

attributes #0 = { nounwind readnone convergent }
