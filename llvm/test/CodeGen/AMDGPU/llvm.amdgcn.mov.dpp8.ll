; RUN: llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefix=GFX10PLUS %s
; RUN: llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefix=GFX10PLUS %s
; RUN: llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx1100 -amdgpu-enable-vopd=0 -verify-machineinstrs < %s | FileCheck -check-prefix=GFX10PLUS %s
; RUN: llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx1100 -amdgpu-enable-vopd=0 -verify-machineinstrs < %s | FileCheck -check-prefix=GFX10PLUS %s

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
; GFX10PLUS: v_mov_b32_dpp v0, v0 dpp8:[1,0,0,0,0,0,0,0]
; GFX10PLUS: v_mov_b32_dpp v1, v1 dpp8:[1,0,0,0,0,0,0,0]
; GFX10PLUS: global_store_{{dwordx2|b64}} v[2:3], v[0:1], off
define amdgpu_ps void @dpp8_i64(i64 %in, ptr addrspace(1) %out) {
  %tmp0 = call i64 @llvm.amdgcn.mov.dpp8.i64(i64 %in, i32 1) #0
  store i64 %tmp0, ptr addrspace(1) %out
  ret void
}

; GFX10PLUS-LABEL: {{^}}dpp8_i128:
; GFX10PLUS: v_mov_b32_dpp v0, v0 dpp8:[1,0,0,0,0,0,0,0]
; GFX10PLUS: v_mov_b32_dpp v1, v1 dpp8:[1,0,0,0,0,0,0,0]
; GFX10PLUS: v_mov_b32_dpp v2, v2 dpp8:[1,0,0,0,0,0,0,0]
; GFX10PLUS: v_mov_b32_dpp v3, v3 dpp8:[1,0,0,0,0,0,0,0]
; GFX10PLUS: global_store_{{dwordx4|b128}} v[4:5], v[0:3], off
define amdgpu_ps void @dpp8_i128(i128 %in, ptr addrspace(1) %out) {
  %tmp0 = call i128 @llvm.amdgcn.mov.dpp8.i128(i128 %in, i32 1) #0
  store i128 %tmp0, ptr addrspace(1) %out
  ret void
}

; GFX10PLUS-LABEL: {{^}}dpp8_i96:
; GFX10PLUS: v_mov_b32_dpp v0, v0 dpp8:[1,0,0,0,0,0,0,0]
; GFX10PLUS: v_mov_b32_dpp v1, v1 dpp8:[1,0,0,0,0,0,0,0]
; GFX10PLUS: v_mov_b32_dpp v2, v2 dpp8:[1,0,0,0,0,0,0,0]
; GFX10PLUS: global_store_{{dwordx3|b96}} v[3:4], v[0:2], off
define amdgpu_ps void @dpp8_i96(i96 %in, ptr addrspace(1) %out) {
  %tmp0 = call i96 @llvm.amdgcn.mov.dpp8.i96(i96 %in, i32 1) #0
  store i96 %tmp0, ptr addrspace(1) %out
  ret void
}

declare i32 @llvm.amdgcn.mov.dpp8.i32(i32, i32) #0

attributes #0 = { nounwind readnone convergent }
