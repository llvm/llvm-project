; RUN: llc -mtriple=amdgcn-- -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SICIVI,FUNC %s
; RUN: llc -mtriple=amdgcn-- -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SICIVI,FUNC %s
; RUN: llc -mtriple=amdgcn-- -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX9,FUNC %s
; RUN: llc -march=r600 -mtriple=r600-- -mcpu=cypress < %s | FileCheck -check-prefixes=EG,FUNC %s

; FUNC-LABEL: {{^}}local_load_i1:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_read_u8
; GCN: v_and_b32_e32 v{{[0-9]+}}, 1
; GCN: ds_write_b8

; EG: LDS_UBYTE_READ_RET
; EG: AND_INT
; EG: LDS_BYTE_WRITE
define amdgpu_kernel void @local_load_i1(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %load = load i1, ptr addrspace(3) %in
  store i1 %load, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_load_v2i1:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0
define amdgpu_kernel void @local_load_v2i1(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %load = load <2 x i1>, ptr addrspace(3) %in
  store <2 x i1> %load, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_load_v3i1:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0
define amdgpu_kernel void @local_load_v3i1(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %load = load <3 x i1>, ptr addrspace(3) %in
  store <3 x i1> %load, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_load_v4i1:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0
define amdgpu_kernel void @local_load_v4i1(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %load = load <4 x i1>, ptr addrspace(3) %in
  store <4 x i1> %load, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_load_v8i1:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0
define amdgpu_kernel void @local_load_v8i1(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %load = load <8 x i1>, ptr addrspace(3) %in
  store <8 x i1> %load, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_load_v16i1:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0
define amdgpu_kernel void @local_load_v16i1(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %load = load <16 x i1>, ptr addrspace(3) %in
  store <16 x i1> %load, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_load_v32i1:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0
define amdgpu_kernel void @local_load_v32i1(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %load = load <32 x i1>, ptr addrspace(3) %in
  store <32 x i1> %load, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_load_v64i1:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0
define amdgpu_kernel void @local_load_v64i1(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %load = load <64 x i1>, ptr addrspace(3) %in
  store <64 x i1> %load, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_i1_to_i32:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_read_u8
; GCN: ds_write_b32
define amdgpu_kernel void @local_zextload_i1_to_i32(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %a = load i1, ptr addrspace(3) %in
  %ext = zext i1 %a to i32
  store i32 %ext, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_i1_to_i32:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_read_u8
; GCN: v_bfe_i32 {{v[0-9]+}}, {{v[0-9]+}}, 0, 1{{$}}
; GCN: ds_write_b32

; EG: LDS_UBYTE_READ_RET
; EG: BFE_INT
define amdgpu_kernel void @local_sextload_i1_to_i32(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %a = load i1, ptr addrspace(3) %in
  %ext = sext i1 %a to i32
  store i32 %ext, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v1i1_to_v1i32:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0
define amdgpu_kernel void @local_zextload_v1i1_to_v1i32(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %load = load <1 x i1>, ptr addrspace(3) %in
  %ext = zext <1 x i1> %load to <1 x i32>
  store <1 x i32> %ext, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v1i1_to_v1i32:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0
define amdgpu_kernel void @local_sextload_v1i1_to_v1i32(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %load = load <1 x i1>, ptr addrspace(3) %in
  %ext = sext <1 x i1> %load to <1 x i32>
  store <1 x i32> %ext, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v2i1_to_v2i32:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0
define amdgpu_kernel void @local_zextload_v2i1_to_v2i32(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %load = load <2 x i1>, ptr addrspace(3) %in
  %ext = zext <2 x i1> %load to <2 x i32>
  store <2 x i32> %ext, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v2i1_to_v2i32:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0
define amdgpu_kernel void @local_sextload_v2i1_to_v2i32(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %load = load <2 x i1>, ptr addrspace(3) %in
  %ext = sext <2 x i1> %load to <2 x i32>
  store <2 x i32> %ext, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v3i1_to_v3i32:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0
define amdgpu_kernel void @local_zextload_v3i1_to_v3i32(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %load = load <3 x i1>, ptr addrspace(3) %in
  %ext = zext <3 x i1> %load to <3 x i32>
  store <3 x i32> %ext, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v3i1_to_v3i32:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0
define amdgpu_kernel void @local_sextload_v3i1_to_v3i32(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %load = load <3 x i1>, ptr addrspace(3) %in
  %ext = sext <3 x i1> %load to <3 x i32>
  store <3 x i32> %ext, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v4i1_to_v4i32:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0
define amdgpu_kernel void @local_zextload_v4i1_to_v4i32(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %load = load <4 x i1>, ptr addrspace(3) %in
  %ext = zext <4 x i1> %load to <4 x i32>
  store <4 x i32> %ext, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v4i1_to_v4i32:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0
define amdgpu_kernel void @local_sextload_v4i1_to_v4i32(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %load = load <4 x i1>, ptr addrspace(3) %in
  %ext = sext <4 x i1> %load to <4 x i32>
  store <4 x i32> %ext, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v8i1_to_v8i32:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0
define amdgpu_kernel void @local_zextload_v8i1_to_v8i32(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %load = load <8 x i1>, ptr addrspace(3) %in
  %ext = zext <8 x i1> %load to <8 x i32>
  store <8 x i32> %ext, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v8i1_to_v8i32:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0
define amdgpu_kernel void @local_sextload_v8i1_to_v8i32(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %load = load <8 x i1>, ptr addrspace(3) %in
  %ext = sext <8 x i1> %load to <8 x i32>
  store <8 x i32> %ext, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v16i1_to_v16i32:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0
define amdgpu_kernel void @local_zextload_v16i1_to_v16i32(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %load = load <16 x i1>, ptr addrspace(3) %in
  %ext = zext <16 x i1> %load to <16 x i32>
  store <16 x i32> %ext, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v16i1_to_v16i32:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0
define amdgpu_kernel void @local_sextload_v16i1_to_v16i32(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %load = load <16 x i1>, ptr addrspace(3) %in
  %ext = sext <16 x i1> %load to <16 x i32>
  store <16 x i32> %ext, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v32i1_to_v32i32:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0
define amdgpu_kernel void @local_zextload_v32i1_to_v32i32(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %load = load <32 x i1>, ptr addrspace(3) %in
  %ext = zext <32 x i1> %load to <32 x i32>
  store <32 x i32> %ext, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v32i1_to_v32i32:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0
define amdgpu_kernel void @local_sextload_v32i1_to_v32i32(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %load = load <32 x i1>, ptr addrspace(3) %in
  %ext = sext <32 x i1> %load to <32 x i32>
  store <32 x i32> %ext, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v64i1_to_v64i32:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0
define amdgpu_kernel void @local_zextload_v64i1_to_v64i32(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %load = load <64 x i1>, ptr addrspace(3) %in
  %ext = zext <64 x i1> %load to <64 x i32>
  store <64 x i32> %ext, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v64i1_to_v64i32:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0
define amdgpu_kernel void @local_sextload_v64i1_to_v64i32(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %load = load <64 x i1>, ptr addrspace(3) %in
  %ext = sext <64 x i1> %load to <64 x i32>
  store <64 x i32> %ext, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_i1_to_i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-DAG: ds_read_u8 [[LOAD:v[0-9]+]],
; GCN-DAG: v_mov_b32_e32 {{v[0-9]+}}, 0{{$}}
; GCN: ds_write_b64
define amdgpu_kernel void @local_zextload_i1_to_i64(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %a = load i1, ptr addrspace(3) %in
  %ext = zext i1 %a to i64
  store i64 %ext, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_i1_to_i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_read_u8 [[LOAD:v[0-9]+]],
; GCN: v_bfe_i32 [[BFE:v[0-9]+]], {{v[0-9]+}}, 0, 1{{$}}
; GCN: v_ashrrev_i32_e32 v{{[0-9]+}}, 31, [[BFE]]
; GCN: ds_write_b64
define amdgpu_kernel void @local_sextload_i1_to_i64(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %a = load i1, ptr addrspace(3) %in
  %ext = sext i1 %a to i64
  store i64 %ext, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v1i1_to_v1i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0
define amdgpu_kernel void @local_zextload_v1i1_to_v1i64(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %load = load <1 x i1>, ptr addrspace(3) %in
  %ext = zext <1 x i1> %load to <1 x i64>
  store <1 x i64> %ext, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v1i1_to_v1i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0
define amdgpu_kernel void @local_sextload_v1i1_to_v1i64(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %load = load <1 x i1>, ptr addrspace(3) %in
  %ext = sext <1 x i1> %load to <1 x i64>
  store <1 x i64> %ext, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v2i1_to_v2i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0
define amdgpu_kernel void @local_zextload_v2i1_to_v2i64(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %load = load <2 x i1>, ptr addrspace(3) %in
  %ext = zext <2 x i1> %load to <2 x i64>
  store <2 x i64> %ext, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v2i1_to_v2i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0
define amdgpu_kernel void @local_sextload_v2i1_to_v2i64(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %load = load <2 x i1>, ptr addrspace(3) %in
  %ext = sext <2 x i1> %load to <2 x i64>
  store <2 x i64> %ext, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v3i1_to_v3i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0
define amdgpu_kernel void @local_zextload_v3i1_to_v3i64(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %load = load <3 x i1>, ptr addrspace(3) %in
  %ext = zext <3 x i1> %load to <3 x i64>
  store <3 x i64> %ext, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v3i1_to_v3i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0
define amdgpu_kernel void @local_sextload_v3i1_to_v3i64(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %load = load <3 x i1>, ptr addrspace(3) %in
  %ext = sext <3 x i1> %load to <3 x i64>
  store <3 x i64> %ext, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v4i1_to_v4i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0
define amdgpu_kernel void @local_zextload_v4i1_to_v4i64(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %load = load <4 x i1>, ptr addrspace(3) %in
  %ext = zext <4 x i1> %load to <4 x i64>
  store <4 x i64> %ext, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v4i1_to_v4i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0
define amdgpu_kernel void @local_sextload_v4i1_to_v4i64(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %load = load <4 x i1>, ptr addrspace(3) %in
  %ext = sext <4 x i1> %load to <4 x i64>
  store <4 x i64> %ext, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v8i1_to_v8i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0
define amdgpu_kernel void @local_zextload_v8i1_to_v8i64(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %load = load <8 x i1>, ptr addrspace(3) %in
  %ext = zext <8 x i1> %load to <8 x i64>
  store <8 x i64> %ext, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v8i1_to_v8i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0
define amdgpu_kernel void @local_sextload_v8i1_to_v8i64(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %load = load <8 x i1>, ptr addrspace(3) %in
  %ext = sext <8 x i1> %load to <8 x i64>
  store <8 x i64> %ext, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v16i1_to_v16i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0
define amdgpu_kernel void @local_zextload_v16i1_to_v16i64(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %load = load <16 x i1>, ptr addrspace(3) %in
  %ext = zext <16 x i1> %load to <16 x i64>
  store <16 x i64> %ext, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v16i1_to_v16i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0
define amdgpu_kernel void @local_sextload_v16i1_to_v16i64(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %load = load <16 x i1>, ptr addrspace(3) %in
  %ext = sext <16 x i1> %load to <16 x i64>
  store <16 x i64> %ext, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v32i1_to_v32i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0
define amdgpu_kernel void @local_zextload_v32i1_to_v32i64(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %load = load <32 x i1>, ptr addrspace(3) %in
  %ext = zext <32 x i1> %load to <32 x i64>
  store <32 x i64> %ext, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v32i1_to_v32i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0
define amdgpu_kernel void @local_sextload_v32i1_to_v32i64(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %load = load <32 x i1>, ptr addrspace(3) %in
  %ext = sext <32 x i1> %load to <32 x i64>
  store <32 x i64> %ext, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v64i1_to_v64i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0
define amdgpu_kernel void @local_zextload_v64i1_to_v64i64(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %load = load <64 x i1>, ptr addrspace(3) %in
  %ext = zext <64 x i1> %load to <64 x i64>
  store <64 x i64> %ext, ptr addrspace(3) %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v64i1_to_v64i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0
define amdgpu_kernel void @local_sextload_v64i1_to_v64i64(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %load = load <64 x i1>, ptr addrspace(3) %in
  %ext = sext <64 x i1> %load to <64 x i64>
  store <64 x i64> %ext, ptr addrspace(3) %out
  ret void
}

attributes #0 = { nounwind }
