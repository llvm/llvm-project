; RUN: llc -mtriple=amdgcn -mtriple=amdgcn-- -mcpu=verde -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SICIVI,FUNC %s
; RUN: llc -mtriple=amdgcn -mtriple=amdgcn-- -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SICIVI,VI,FUNC %s
; RUN: llc -mtriple=amdgcn -mtriple=amdgcn-- -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX9,FUNC %s
; RUN: llc -march=r600 -mtriple=r600-- -mcpu=redwood < %s | FileCheck -check-prefixes=EG,FUNC %s
; RUN: llc -march=r600 -mtriple=r600-- -mcpu=cayman < %s | FileCheck -check-prefixes=CM,FUNC %s

; FUNC-LABEL: {{^}}store_local_i1:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; EG: LDS_BYTE_WRITE

; CM: LDS_BYTE_WRITE

; GCN: ds_write_b8
define amdgpu_kernel void @store_local_i1(ptr addrspace(3) %out) {
entry:
  store i1 true, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}store_local_i8:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; EG: LDS_BYTE_WRITE

; CM: LDS_BYTE_WRITE

; GCN: ds_write_b8
define amdgpu_kernel void @store_local_i8(ptr addrspace(3) %out, i8 %in) {
  store i8 %in, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}store_local_i16:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; EG: LDS_SHORT_WRITE

; CM: LDS_SHORT_WRITE

; GCN: ds_write_b16
define amdgpu_kernel void @store_local_i16(ptr addrspace(3) %out, i16 %in) {
  store i16 %in, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}store_local_v2i16:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; EG: LDS_WRITE

; CM: LDS_WRITE

; GCN: ds_write_b32
define amdgpu_kernel void @store_local_v2i16(ptr addrspace(3) %out, <2 x i16> %in) {
entry:
  store <2 x i16> %in, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}store_local_v4i8:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; EG: LDS_WRITE

; CM: LDS_WRITE

; GCN: ds_write_b32
define amdgpu_kernel void @store_local_v4i8(ptr addrspace(3) %out, <4 x i8> %in) {
entry:
  store <4 x i8> %in, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}store_local_v4i8_unaligned:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; EG: LDS_BYTE_WRITE
; EG: LDS_BYTE_WRITE
; EG: LDS_BYTE_WRITE
; EG: LDS_BYTE_WRITE
; EG-NOT: LDS_WRITE

; CM: LDS_BYTE_WRITE
; CM: LDS_BYTE_WRITE
; CM: LDS_BYTE_WRITE
; CM: LDS_BYTE_WRITE
; CM-NOT: LDS_WRITE

; GCN: ds_write_b8
; GCN: ds_write_b8
; GCN: ds_write_b8
; GCN: ds_write_b8
define amdgpu_kernel void @store_local_v4i8_unaligned(ptr addrspace(3) %out, <4 x i8> %in) {
entry:
  store <4 x i8> %in, ptr addrspace(3) %out, align 1
  ret void
}

; FUNC-LABEL: {{^}}store_local_v4i8_halfaligned:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; EG: LDS_SHORT_WRITE
; EG: LDS_SHORT_WRITE
; EG-NOT: LDS_WRITE

; CM: LDS_SHORT_WRITE
; CM: LDS_SHORT_WRITE
; CM-NOT: LDS_WRITE

; GCN: ds_write_b16
; GCN: ds_write_b16
define amdgpu_kernel void @store_local_v4i8_halfaligned(ptr addrspace(3) %out, <4 x i8> %in) {
entry:
  store <4 x i8> %in, ptr addrspace(3) %out, align 2
  ret void
}

; FUNC-LABEL: {{^}}store_local_v2i32:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; EG: LDS_WRITE
; EG: LDS_WRITE
; EG-NOT: LDS_WRITE

; CM: LDS_WRITE
; CM: LDS_WRITE
; CM-NOT: LDS_WRITE

; GCN: ds_write_b64
define amdgpu_kernel void @store_local_v2i32(ptr addrspace(3) %out, <2 x i32> %in) {
entry:
  store <2 x i32> %in, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}store_local_v4i32:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE

; CM: LDS_WRITE
; CM: LDS_WRITE
; CM: LDS_WRITE
; CM: LDS_WRITE

; SI: ds_write2_b32
; VI: ds_write_b128
; GFX9: ds_write_b128
define amdgpu_kernel void @store_local_v4i32(ptr addrspace(3) %out, <4 x i32> %in) {
entry:
  store <4 x i32> %in, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}store_local_v4i32_align4:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE

; CM: LDS_WRITE
; CM: LDS_WRITE
; CM: LDS_WRITE
; CM: LDS_WRITE

; GCN: ds_write2_b32
; GCN: ds_write2_b32
define amdgpu_kernel void @store_local_v4i32_align4(ptr addrspace(3) %out, <4 x i32> %in) {
entry:
  store <4 x i32> %in, ptr addrspace(3) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}store_local_i64_i8:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; EG: LDS_BYTE_WRITE
; GCN: ds_write_b8
define amdgpu_kernel void @store_local_i64_i8(ptr addrspace(3) %out, i64 %in) {
entry:
  %0 = trunc i64 %in to i8
  store i8 %0, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}store_local_i64_i16:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; EG: LDS_SHORT_WRITE
; GCN: ds_write_b16
define amdgpu_kernel void @store_local_i64_i16(ptr addrspace(3) %out, i64 %in) {
entry:
  %0 = trunc i64 %in to i16
  store i16 %0, ptr addrspace(3) %out
  ret void
}
