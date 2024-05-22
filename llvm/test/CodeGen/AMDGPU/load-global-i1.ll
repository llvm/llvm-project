; RUN: llc -mtriple=amdgcn-- -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=FUNC %s
; RUN: llc -mtriple=amdgcn-- -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=FUNC %s
; RUN: llc -march=r600 -mtriple=r600-- -mcpu=cypress < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}global_load_i1:
; GCN: buffer_load_ubyte
; GCN: v_and_b32_e32 v{{[0-9]+}}, 1
; GCN: buffer_store_byte

; EG: VTX_READ_8
; EG: AND_INT
define amdgpu_kernel void @global_load_i1(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load i1, ptr addrspace(1) %in
  store i1 %load, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v2i1:
define amdgpu_kernel void @global_load_v2i1(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <2 x i1>, ptr addrspace(1) %in
  store <2 x i1> %load, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v3i1:
define amdgpu_kernel void @global_load_v3i1(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <3 x i1>, ptr addrspace(1) %in
  store <3 x i1> %load, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v4i1:
define amdgpu_kernel void @global_load_v4i1(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <4 x i1>, ptr addrspace(1) %in
  store <4 x i1> %load, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v8i1:
define amdgpu_kernel void @global_load_v8i1(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <8 x i1>, ptr addrspace(1) %in
  store <8 x i1> %load, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v16i1:
define amdgpu_kernel void @global_load_v16i1(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <16 x i1>, ptr addrspace(1) %in
  store <16 x i1> %load, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v32i1:
define amdgpu_kernel void @global_load_v32i1(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <32 x i1>, ptr addrspace(1) %in
  store <32 x i1> %load, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v64i1:
define amdgpu_kernel void @global_load_v64i1(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <64 x i1>, ptr addrspace(1) %in
  store <64 x i1> %load, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_i1_to_i32:
; GCN: buffer_load_ubyte
; GCN: buffer_store_dword
define amdgpu_kernel void @global_zextload_i1_to_i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %a = load i1, ptr addrspace(1) %in
  %ext = zext i1 %a to i32
  store i32 %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_i1_to_i32:
; GCN: buffer_load_ubyte
; GCN: v_bfe_i32 {{v[0-9]+}}, {{v[0-9]+}}, 0, 1{{$}}
; GCN: buffer_store_dword

; EG: VTX_READ_8
; EG: BFE_INT
define amdgpu_kernel void @global_sextload_i1_to_i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %a = load i1, ptr addrspace(1) %in
  %ext = sext i1 %a to i32
  store i32 %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v1i1_to_v1i32:
define amdgpu_kernel void @global_zextload_v1i1_to_v1i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <1 x i1>, ptr addrspace(1) %in
  %ext = zext <1 x i1> %load to <1 x i32>
  store <1 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v1i1_to_v1i32:
define amdgpu_kernel void @global_sextload_v1i1_to_v1i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <1 x i1>, ptr addrspace(1) %in
  %ext = sext <1 x i1> %load to <1 x i32>
  store <1 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v2i1_to_v2i32:
define amdgpu_kernel void @global_zextload_v2i1_to_v2i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <2 x i1>, ptr addrspace(1) %in
  %ext = zext <2 x i1> %load to <2 x i32>
  store <2 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v2i1_to_v2i32:
define amdgpu_kernel void @global_sextload_v2i1_to_v2i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <2 x i1>, ptr addrspace(1) %in
  %ext = sext <2 x i1> %load to <2 x i32>
  store <2 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v3i1_to_v3i32:
define amdgpu_kernel void @global_zextload_v3i1_to_v3i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <3 x i1>, ptr addrspace(1) %in
  %ext = zext <3 x i1> %load to <3 x i32>
  store <3 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v3i1_to_v3i32:
define amdgpu_kernel void @global_sextload_v3i1_to_v3i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <3 x i1>, ptr addrspace(1) %in
  %ext = sext <3 x i1> %load to <3 x i32>
  store <3 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v4i1_to_v4i32:
define amdgpu_kernel void @global_zextload_v4i1_to_v4i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <4 x i1>, ptr addrspace(1) %in
  %ext = zext <4 x i1> %load to <4 x i32>
  store <4 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v4i1_to_v4i32:
define amdgpu_kernel void @global_sextload_v4i1_to_v4i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <4 x i1>, ptr addrspace(1) %in
  %ext = sext <4 x i1> %load to <4 x i32>
  store <4 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v8i1_to_v8i32:
define amdgpu_kernel void @global_zextload_v8i1_to_v8i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <8 x i1>, ptr addrspace(1) %in
  %ext = zext <8 x i1> %load to <8 x i32>
  store <8 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v8i1_to_v8i32:
define amdgpu_kernel void @global_sextload_v8i1_to_v8i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <8 x i1>, ptr addrspace(1) %in
  %ext = sext <8 x i1> %load to <8 x i32>
  store <8 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v16i1_to_v16i32:
define amdgpu_kernel void @global_zextload_v16i1_to_v16i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <16 x i1>, ptr addrspace(1) %in
  %ext = zext <16 x i1> %load to <16 x i32>
  store <16 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v16i1_to_v16i32:
define amdgpu_kernel void @global_sextload_v16i1_to_v16i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <16 x i1>, ptr addrspace(1) %in
  %ext = sext <16 x i1> %load to <16 x i32>
  store <16 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v32i1_to_v32i32:
define amdgpu_kernel void @global_zextload_v32i1_to_v32i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <32 x i1>, ptr addrspace(1) %in
  %ext = zext <32 x i1> %load to <32 x i32>
  store <32 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v32i1_to_v32i32:
define amdgpu_kernel void @global_sextload_v32i1_to_v32i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <32 x i1>, ptr addrspace(1) %in
  %ext = sext <32 x i1> %load to <32 x i32>
  store <32 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v64i1_to_v64i32:
define amdgpu_kernel void @global_zextload_v64i1_to_v64i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <64 x i1>, ptr addrspace(1) %in
  %ext = zext <64 x i1> %load to <64 x i32>
  store <64 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v64i1_to_v64i32:
define amdgpu_kernel void @global_sextload_v64i1_to_v64i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <64 x i1>, ptr addrspace(1) %in
  %ext = sext <64 x i1> %load to <64 x i32>
  store <64 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_i1_to_i64:
; GCN-DAG: buffer_load_ubyte [[LOAD:v[0-9]+]],
; GCN-DAG: v_mov_b32_e32 {{v[0-9]+}}, 0{{$}}
; GCN-DAG: v_and_b32_e32 {{v[0-9]+}}, 1, [[LOAD]]{{$}}
; GCN: buffer_store_dwordx2
define amdgpu_kernel void @global_zextload_i1_to_i64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %a = load i1, ptr addrspace(1) %in
  %ext = zext i1 %a to i64
  store i64 %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_i1_to_i64:
; GCN: buffer_load_ubyte [[LOAD:v[0-9]+]],
; GCN: v_bfe_i32 [[BFE:v[0-9]+]], {{v[0-9]+}}, 0, 1{{$}}
; GCN: v_ashrrev_i32_e32 v{{[0-9]+}}, 31, [[BFE]]
; GCN: buffer_store_dwordx2
define amdgpu_kernel void @global_sextload_i1_to_i64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %a = load i1, ptr addrspace(1) %in
  %ext = sext i1 %a to i64
  store i64 %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v1i1_to_v1i64:
define amdgpu_kernel void @global_zextload_v1i1_to_v1i64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <1 x i1>, ptr addrspace(1) %in
  %ext = zext <1 x i1> %load to <1 x i64>
  store <1 x i64> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v1i1_to_v1i64:
define amdgpu_kernel void @global_sextload_v1i1_to_v1i64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <1 x i1>, ptr addrspace(1) %in
  %ext = sext <1 x i1> %load to <1 x i64>
  store <1 x i64> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v2i1_to_v2i64:
define amdgpu_kernel void @global_zextload_v2i1_to_v2i64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <2 x i1>, ptr addrspace(1) %in
  %ext = zext <2 x i1> %load to <2 x i64>
  store <2 x i64> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v2i1_to_v2i64:
define amdgpu_kernel void @global_sextload_v2i1_to_v2i64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <2 x i1>, ptr addrspace(1) %in
  %ext = sext <2 x i1> %load to <2 x i64>
  store <2 x i64> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v3i1_to_v3i64:
define amdgpu_kernel void @global_zextload_v3i1_to_v3i64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <3 x i1>, ptr addrspace(1) %in
  %ext = zext <3 x i1> %load to <3 x i64>
  store <3 x i64> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v3i1_to_v3i64:
define amdgpu_kernel void @global_sextload_v3i1_to_v3i64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <3 x i1>, ptr addrspace(1) %in
  %ext = sext <3 x i1> %load to <3 x i64>
  store <3 x i64> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v4i1_to_v4i64:
define amdgpu_kernel void @global_zextload_v4i1_to_v4i64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <4 x i1>, ptr addrspace(1) %in
  %ext = zext <4 x i1> %load to <4 x i64>
  store <4 x i64> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v4i1_to_v4i64:
define amdgpu_kernel void @global_sextload_v4i1_to_v4i64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <4 x i1>, ptr addrspace(1) %in
  %ext = sext <4 x i1> %load to <4 x i64>
  store <4 x i64> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v8i1_to_v8i64:
define amdgpu_kernel void @global_zextload_v8i1_to_v8i64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <8 x i1>, ptr addrspace(1) %in
  %ext = zext <8 x i1> %load to <8 x i64>
  store <8 x i64> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v8i1_to_v8i64:
define amdgpu_kernel void @global_sextload_v8i1_to_v8i64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <8 x i1>, ptr addrspace(1) %in
  %ext = sext <8 x i1> %load to <8 x i64>
  store <8 x i64> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v16i1_to_v16i64:
define amdgpu_kernel void @global_zextload_v16i1_to_v16i64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <16 x i1>, ptr addrspace(1) %in
  %ext = zext <16 x i1> %load to <16 x i64>
  store <16 x i64> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v16i1_to_v16i64:
define amdgpu_kernel void @global_sextload_v16i1_to_v16i64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <16 x i1>, ptr addrspace(1) %in
  %ext = sext <16 x i1> %load to <16 x i64>
  store <16 x i64> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v32i1_to_v32i64:
define amdgpu_kernel void @global_zextload_v32i1_to_v32i64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <32 x i1>, ptr addrspace(1) %in
  %ext = zext <32 x i1> %load to <32 x i64>
  store <32 x i64> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v32i1_to_v32i64:
define amdgpu_kernel void @global_sextload_v32i1_to_v32i64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <32 x i1>, ptr addrspace(1) %in
  %ext = sext <32 x i1> %load to <32 x i64>
  store <32 x i64> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v64i1_to_v64i64:
define amdgpu_kernel void @global_zextload_v64i1_to_v64i64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <64 x i1>, ptr addrspace(1) %in
  %ext = zext <64 x i1> %load to <64 x i64>
  store <64 x i64> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v64i1_to_v64i64:
define amdgpu_kernel void @global_sextload_v64i1_to_v64i64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <64 x i1>, ptr addrspace(1) %in
  %ext = sext <64 x i1> %load to <64 x i64>
  store <64 x i64> %ext, ptr addrspace(1) %out
  ret void
}

attributes #0 = { nounwind }
