; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX9_11 %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX10 %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX9_11 %s

; GCN-LABEL: flat_inst_offset:
; GFX9_11: flat_load_{{dword|b32}} v{{[0-9]+}}, v[{{[0-9:]+}}] offset:4
; GFX9_11: flat_store_{{dword|b32}} v[{{[0-9:]+}}], v{{[0-9]+}} offset:4
; GFX10: flat_load_dword v{{[0-9]+}}, v[{{[0-9:]+}}]{{$}}
; GFX10: flat_store_dword v[{{[0-9:]+}}], v{{[0-9]+}}{{$}}
define void @flat_inst_offset(ptr nocapture %p) {
  %gep = getelementptr inbounds i32, ptr %p, i64 1
  %load = load i32, ptr %gep, align 4
  %inc = add nsw i32 %load, 1
  store i32 %inc, ptr %gep, align 4
  ret void
}

; GCN-LABEL: global_inst_offset:
; GCN: global_load_{{dword|b32}} v{{[0-9]+}}, v[{{[0-9:]+}}], off offset:4
; GCN: global_store_{{dword|b32}} v[{{[0-9:]+}}], v{{[0-9]+}}, off offset:4
define void @global_inst_offset(ptr addrspace(1) nocapture %p) {
  %gep = getelementptr inbounds i32, ptr addrspace(1) %p, i64 1
  %load = load i32, ptr addrspace(1) %gep, align 4
  %inc = add nsw i32 %load, 1
  store i32 %inc, ptr addrspace(1) %gep, align 4
  ret void
}

; GCN-LABEL: load_i16_lo:
; GFX9_11: flat_load_{{short_d16|d16_b16}} v{{[0-9]+}}, v[{{[0-9:]+}}] offset:8{{$}}
; GFX10: flat_load_short_d16 v{{[0-9]+}}, v[{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @load_i16_lo(ptr %arg, ptr %out) {
  %gep = getelementptr inbounds i16, ptr %arg, i32 4
  %ld = load i16, ptr %gep, align 2
  %vec = insertelement <2 x i16> <i16 undef, i16 0>, i16 %ld, i32 0
  %v = add <2 x i16> %vec, %vec
  store <2 x i16> %v, ptr %out, align 4
  ret void
}

; GCN-LABEL: load_i16_hi:
; GFX9_11: flat_load_{{short_d16_hi|d16_hi_b16}} v{{[0-9]+}}, v[{{[0-9:]+}}] offset:8{{$}}
; GFX10: flat_load_short_d16_hi v{{[0-9]+}}, v[{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @load_i16_hi(ptr %arg, ptr %out) {
  %gep = getelementptr inbounds i16, ptr %arg, i32 4
  %ld = load i16, ptr %gep, align 2
  %vec = insertelement <2 x i16> <i16 0, i16 undef>, i16 %ld, i32 1
  %v = add <2 x i16> %vec, %vec
  store <2 x i16> %v, ptr %out, align 4
  ret void
}

; GCN-LABEL: load_half_lo:
; GFX9_11: flat_load_{{short_d16|d16_b16}} v{{[0-9]+}}, v[{{[0-9:]+}}] offset:8{{$}}
; GFX10: flat_load_short_d16 v{{[0-9]+}}, v[{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @load_half_lo(ptr %arg, ptr %out) {
  %gep = getelementptr inbounds half, ptr %arg, i32 4
  %ld = load half, ptr %gep, align 2
  %vec = insertelement <2 x half> <half undef, half 0xH0000>, half %ld, i32 0
  %v = fadd <2 x half> %vec, %vec
  store <2 x half> %v, ptr %out, align 4
  ret void
}

; GCN-LABEL: load_half_hi:
; GFX9_11: flat_load_{{short_d16_hi|d16_hi_b16}} v{{[0-9]+}}, v[{{[0-9:]+}}] offset:8{{$}}
; GFX10: flat_load_short_d16_hi v{{[0-9]+}}, v[{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @load_half_hi(ptr %arg, ptr %out) {
  %gep = getelementptr inbounds half, ptr %arg, i32 4
  %ld = load half, ptr %gep, align 2
  %vec = insertelement <2 x half> <half 0xH0000, half undef>, half %ld, i32 1
  %v = fadd <2 x half> %vec, %vec
  store <2 x half> %v, ptr %out, align 4
  ret void
}

; GCN-LABEL: load_float_lo:
; GFX9_11: flat_load_{{dword|b32}} v{{[0-9]+}}, v[{{[0-9:]+}}] offset:16{{$}}
; GFX10: flat_load_dword v{{[0-9]+}}, v[{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @load_float_lo(ptr %arg, ptr %out) {
  %gep = getelementptr inbounds float, ptr %arg, i32 4
  %ld = load float, ptr %gep, align 4
  %v = fadd float %ld, %ld
  store float %v, ptr %out, align 4
  ret void
}
