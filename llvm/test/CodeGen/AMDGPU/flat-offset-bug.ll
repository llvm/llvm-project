; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX9 %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX10 %s

; GCN-LABEL: flat_inst_offset:
; GFX9:  flat_load_dword v{{[0-9]+}}, v[{{[0-9:]+}}] offset:4
; GFX9:  flat_store_dword v[{{[0-9:]+}}], v{{[0-9]+}} offset:4
; GFX10: flat_load_dword v{{[0-9]+}}, v[{{[0-9:]+}}]{{$}}
; GFX10: flat_store_dword v[{{[0-9:]+}}], v{{[0-9]+}}{{$}}
define void @flat_inst_offset(i32* nocapture %p) {
  %gep = getelementptr inbounds i32, i32* %p, i64 1
  %load = load i32, i32* %gep, align 4
  %inc = add nsw i32 %load, 1
  store i32 %inc, i32* %gep, align 4
  ret void
}

; GCN-LABEL: global_inst_offset:
; GCN: global_load_dword v{{[0-9]+}}, v[{{[0-9:]+}}], off offset:4
; GCN: global_store_dword v[{{[0-9:]+}}], v{{[0-9]+}}, off offset:4
define void @global_inst_offset(i32 addrspace(1)* nocapture %p) {
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %p, i64 1
  %load = load i32, i32 addrspace(1)* %gep, align 4
  %inc = add nsw i32 %load, 1
  store i32 %inc, i32 addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: load_i16_lo:
; GFX9:  flat_load_short_d16 v{{[0-9]+}}, v[{{[0-9:]+}}] offset:8{{$}}
; GFX10: flat_load_short_d16 v{{[0-9]+}}, v[{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @load_i16_lo(i16* %arg, <2 x i16>* %out) {
  %gep = getelementptr inbounds i16, i16* %arg, i32 4
  %ld = load i16, i16* %gep, align 2
  %vec = insertelement <2 x i16> <i16 undef, i16 0>, i16 %ld, i32 0
  %v = add <2 x i16> %vec, %vec
  store <2 x i16> %v, <2 x i16>* %out, align 4
  ret void
}

; GCN-LABEL: load_i16_hi:
; GFX9:  flat_load_short_d16_hi v{{[0-9]+}}, v[{{[0-9:]+}}] offset:8{{$}}
; GFX10: flat_load_short_d16_hi v{{[0-9]+}}, v[{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @load_i16_hi(i16* %arg, <2 x i16>* %out) {
  %gep = getelementptr inbounds i16, i16* %arg, i32 4
  %ld = load i16, i16* %gep, align 2
  %vec = insertelement <2 x i16> <i16 0, i16 undef>, i16 %ld, i32 1
  %v = add <2 x i16> %vec, %vec
  store <2 x i16> %v, <2 x i16>* %out, align 4
  ret void
}

; GCN-LABEL: load_half_lo:
; GFX9:  flat_load_short_d16 v{{[0-9]+}}, v[{{[0-9:]+}}] offset:8{{$}}
; GFX10: flat_load_short_d16 v{{[0-9]+}}, v[{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @load_half_lo(half* %arg, <2 x half>* %out) {
  %gep = getelementptr inbounds half, half* %arg, i32 4
  %ld = load half, half* %gep, align 2
  %vec = insertelement <2 x half> <half undef, half 0xH0000>, half %ld, i32 0
  %v = fadd <2 x half> %vec, %vec
  store <2 x half> %v, <2 x half>* %out, align 4
  ret void
}

; GCN-LABEL: load_half_hi:
; GFX9:  flat_load_short_d16_hi v{{[0-9]+}}, v[{{[0-9:]+}}] offset:8{{$}}
; GFX10: flat_load_short_d16_hi v{{[0-9]+}}, v[{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @load_half_hi(half* %arg, <2 x half>* %out) {
  %gep = getelementptr inbounds half, half* %arg, i32 4
  %ld = load half, half* %gep, align 2
  %vec = insertelement <2 x half> <half 0xH0000, half undef>, half %ld, i32 1
  %v = fadd <2 x half> %vec, %vec
  store <2 x half> %v, <2 x half>* %out, align 4
  ret void
}

; GCN-LABEL: load_float_lo:
; GFX9:  flat_load_dword v{{[0-9]+}}, v[{{[0-9:]+}}] offset:16{{$}}
; GFX10: flat_load_dword v{{[0-9]+}}, v[{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @load_float_lo(float* %arg, float* %out) {
  %gep = getelementptr inbounds float, float* %arg, i32 4
  %ld = load float, float* %gep, align 4
  %v = fadd float %ld, %ld
  store float %v, float* %out, align 4
  ret void
}
