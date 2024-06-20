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

declare i32 @llvm.amdgcn.mov.dpp8.i32(i32, i32) #0

attributes #0 = { nounwind readnone convergent }
