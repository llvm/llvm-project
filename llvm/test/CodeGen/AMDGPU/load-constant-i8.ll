; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GCN-NOHSA -check-prefix=FUNC %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GCN-HSA -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GCN-NOHSA -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s


; FUNC-LABEL: {{^}}constant_load_i8:
; GCN-NOHSA: buffer_load_ubyte v{{[0-9]+}}
; GCN-HSA: flat_load_ubyte

; EG: VTX_READ_8 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0
define void @constant_load_i8(i8 addrspace(1)* %out, i8 addrspace(2)* %in) #0 {
entry:
  %ld = load i8, i8 addrspace(2)* %in
  store i8 %ld, i8 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_load_v2i8:
; GCN-NOHSA: buffer_load_ushort v
; GCN-HSA: flat_load_ushort v

; EG: VTX_READ_16
define void @constant_load_v2i8(<2 x i8> addrspace(1)* %out, <2 x i8> addrspace(2)* %in) #0 {
entry:
  %ld = load <2 x i8>, <2 x i8> addrspace(2)* %in
  store <2 x i8> %ld, <2 x i8> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_load_v3i8:
; GCN: s_load_dword s

; EG-DAG: VTX_READ_32
define void @constant_load_v3i8(<3 x i8> addrspace(1)* %out, <3 x i8> addrspace(2)* %in) #0 {
entry:
  %ld = load <3 x i8>, <3 x i8> addrspace(2)* %in
  store <3 x i8> %ld, <3 x i8> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_load_v4i8:
; GCN: s_load_dword s

; EG: VTX_READ_32
define void @constant_load_v4i8(<4 x i8> addrspace(1)* %out, <4 x i8> addrspace(2)* %in) #0 {
entry:
  %ld = load <4 x i8>, <4 x i8> addrspace(2)* %in
  store <4 x i8> %ld, <4 x i8> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_load_v8i8:
; GCN: s_load_dwordx2

; EG: VTX_READ_64
define void @constant_load_v8i8(<8 x i8> addrspace(1)* %out, <8 x i8> addrspace(2)* %in) #0 {
entry:
  %ld = load <8 x i8>, <8 x i8> addrspace(2)* %in
  store <8 x i8> %ld, <8 x i8> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_load_v16i8:
; GCN: s_load_dwordx4

; EG: VTX_READ_128
define void @constant_load_v16i8(<16 x i8> addrspace(1)* %out, <16 x i8> addrspace(2)* %in) #0 {
entry:
  %ld = load <16 x i8>, <16 x i8> addrspace(2)* %in
  store <16 x i8> %ld, <16 x i8> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_zextload_i8_to_i32:
; GCN-NOHSA: buffer_load_ubyte v{{[0-9]+}},
; GCN-HSA: flat_load_ubyte

; EG: VTX_READ_8 T{{[0-9]+\.X, T[0-9]+\.X}}
define void @constant_zextload_i8_to_i32(i32 addrspace(1)* %out, i8 addrspace(2)* %in) #0 {
  %a = load i8, i8 addrspace(2)* %in
  %ext = zext i8 %a to i32
  store i32 %ext, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_sextload_i8_to_i32:
; GCN-NOHSA: buffer_load_sbyte
; GCN-HSA: flat_load_sbyte

; EG: VTX_READ_8 [[DST:T[0-9]\.[XYZW]]], [[DST]]
; EG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, [[DST]], 0.0, literal
; EG: 8
define void @constant_sextload_i8_to_i32(i32 addrspace(1)* %out, i8 addrspace(2)* %in) #0 {
  %ld = load i8, i8 addrspace(2)* %in
  %ext = sext i8 %ld to i32
  store i32 %ext, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_zextload_v1i8_to_v1i32:
define void @constant_zextload_v1i8_to_v1i32(<1 x i32> addrspace(1)* %out, <1 x i8> addrspace(2)* %in) #0 {
  %load = load <1 x i8>, <1 x i8> addrspace(2)* %in
  %ext = zext <1 x i8> %load to <1 x i32>
  store <1 x i32> %ext, <1 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_sextload_v1i8_to_v1i32:
define void @constant_sextload_v1i8_to_v1i32(<1 x i32> addrspace(1)* %out, <1 x i8> addrspace(2)* %in) #0 {
  %load = load <1 x i8>, <1 x i8> addrspace(2)* %in
  %ext = sext <1 x i8> %load to <1 x i32>
  store <1 x i32> %ext, <1 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_zextload_v2i8_to_v2i32:
; GCN-NOHSA: buffer_load_ushort
; GCN-HSA: flat_load_ushort
; EG: VTX_READ_8
; EG: VTX_READ_8
define void @constant_zextload_v2i8_to_v2i32(<2 x i32> addrspace(1)* %out, <2 x i8> addrspace(2)* %in) #0 {
  %load = load <2 x i8>, <2 x i8> addrspace(2)* %in
  %ext = zext <2 x i8> %load to <2 x i32>
  store <2 x i32> %ext, <2 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_sextload_v2i8_to_v2i32:
; GCN-NOHSA: buffer_load_ushort

; GCN-HSA: flat_load_ushort

; GCN: v_bfe_i32
; GCN: v_bfe_i32

; EG-DAG: VTX_READ_8 [[DST_X:T[0-9]\.[XYZW]]], [[DST_X]]
; EG-DAG: VTX_READ_8 [[DST_Y:T[0-9]\.[XYZW]]], [[DST_Y]]
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, [[DST_X]], 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, [[DST_Y]], 0.0, literal
; EG-DAG: 8
; EG-DAG: 8
define void @constant_sextload_v2i8_to_v2i32(<2 x i32> addrspace(1)* %out, <2 x i8> addrspace(2)* %in) #0 {
  %load = load <2 x i8>, <2 x i8> addrspace(2)* %in
  %ext = sext <2 x i8> %load to <2 x i32>
  store <2 x i32> %ext, <2 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_zextload_v3i8_to_v3i32:
; GCN: s_load_dword s

; GCN-DAG: s_bfe_u32
; GCN-DAG: s_bfe_u32
; GCN-DAG: s_and_b32
define void @constant_zextload_v3i8_to_v3i32(<3 x i32> addrspace(1)* %out, <3 x i8> addrspace(2)* %in) #0 {
entry:
  %ld = load <3 x i8>, <3 x i8> addrspace(2)* %in
  %ext = zext <3 x i8> %ld to <3 x i32>
  store <3 x i32> %ext, <3 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_sextload_v3i8_to_v3i32:
; GCN: s_load_dword s

; GCN-DAG: s_bfe_i32
; GCN-DAG: s_bfe_i32
; GCN-DAG: s_bfe_i32
define void @constant_sextload_v3i8_to_v3i32(<3 x i32> addrspace(1)* %out, <3 x i8> addrspace(2)* %in) #0 {
entry:
  %ld = load <3 x i8>, <3 x i8> addrspace(2)* %in
  %ext = sext <3 x i8> %ld to <3 x i32>
  store <3 x i32> %ext, <3 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_zextload_v4i8_to_v4i32:
; GCN: s_load_dword s
; GCN-DAG: s_and_b32
; GCN-DAG: s_lshr_b32

; EG: VTX_READ_8
; EG: VTX_READ_8
; EG: VTX_READ_8
; EG: VTX_READ_8
define void @constant_zextload_v4i8_to_v4i32(<4 x i32> addrspace(1)* %out, <4 x i8> addrspace(2)* %in) #0 {
  %load = load <4 x i8>, <4 x i8> addrspace(2)* %in
  %ext = zext <4 x i8> %load to <4 x i32>
  store <4 x i32> %ext, <4 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_sextload_v4i8_to_v4i32:
; GCN: s_load_dword s
; GCN-DAG: s_sext_i32_i8
; GCN-DAG: s_ashr_i32

; EG-DAG: VTX_READ_8 [[DST_X:T[0-9]\.[XYZW]]], [[DST_X]]
; EG-DAG: VTX_READ_8 [[DST_Y:T[0-9]\.[XYZW]]], [[DST_Y]]
; EG-DAG: VTX_READ_8 [[DST_Z:T[0-9]\.[XYZW]]], [[DST_Z]]
; EG-DAG: VTX_READ_8 [[DST_W:T[0-9]\.[XYZW]]], [[DST_W]]
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, [[DST_X]], 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, [[DST_Y]], 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, [[DST_Z]], 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, [[DST_W]], 0.0, literal
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
define void @constant_sextload_v4i8_to_v4i32(<4 x i32> addrspace(1)* %out, <4 x i8> addrspace(2)* %in) #0 {
  %load = load <4 x i8>, <4 x i8> addrspace(2)* %in
  %ext = sext <4 x i8> %load to <4 x i32>
  store <4 x i32> %ext, <4 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_zextload_v8i8_to_v8i32:
; GCN: s_load_dwordx2
; GCN-DAG: s_and_b32
; GCN-DAG: s_lshr_b32
define void @constant_zextload_v8i8_to_v8i32(<8 x i32> addrspace(1)* %out, <8 x i8> addrspace(2)* %in) #0 {
  %load = load <8 x i8>, <8 x i8> addrspace(2)* %in
  %ext = zext <8 x i8> %load to <8 x i32>
  store <8 x i32> %ext, <8 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_sextload_v8i8_to_v8i32:
; GCN: s_load_dwordx2
; GCN-DAG: s_ashr_i32
; GCN-DAG: s_sext_i32_i8
define void @constant_sextload_v8i8_to_v8i32(<8 x i32> addrspace(1)* %out, <8 x i8> addrspace(2)* %in) #0 {
  %load = load <8 x i8>, <8 x i8> addrspace(2)* %in
  %ext = sext <8 x i8> %load to <8 x i32>
  store <8 x i32> %ext, <8 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_zextload_v16i8_to_v16i32:
define void @constant_zextload_v16i8_to_v16i32(<16 x i32> addrspace(1)* %out, <16 x i8> addrspace(2)* %in) #0 {
  %load = load <16 x i8>, <16 x i8> addrspace(2)* %in
  %ext = zext <16 x i8> %load to <16 x i32>
  store <16 x i32> %ext, <16 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_sextload_v16i8_to_v16i32:
define void @constant_sextload_v16i8_to_v16i32(<16 x i32> addrspace(1)* %out, <16 x i8> addrspace(2)* %in) #0 {
  %load = load <16 x i8>, <16 x i8> addrspace(2)* %in
  %ext = sext <16 x i8> %load to <16 x i32>
  store <16 x i32> %ext, <16 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_zextload_v32i8_to_v32i32:
define void @constant_zextload_v32i8_to_v32i32(<32 x i32> addrspace(1)* %out, <32 x i8> addrspace(2)* %in) #0 {
  %load = load <32 x i8>, <32 x i8> addrspace(2)* %in
  %ext = zext <32 x i8> %load to <32 x i32>
  store <32 x i32> %ext, <32 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_sextload_v32i8_to_v32i32:
define void @constant_sextload_v32i8_to_v32i32(<32 x i32> addrspace(1)* %out, <32 x i8> addrspace(2)* %in) #0 {
  %load = load <32 x i8>, <32 x i8> addrspace(2)* %in
  %ext = sext <32 x i8> %load to <32 x i32>
  store <32 x i32> %ext, <32 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_zextload_v64i8_to_v64i32:
define void @constant_zextload_v64i8_to_v64i32(<64 x i32> addrspace(1)* %out, <64 x i8> addrspace(2)* %in) #0 {
  %load = load <64 x i8>, <64 x i8> addrspace(2)* %in
  %ext = zext <64 x i8> %load to <64 x i32>
  store <64 x i32> %ext, <64 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_sextload_v64i8_to_v64i32:
define void @constant_sextload_v64i8_to_v64i32(<64 x i32> addrspace(1)* %out, <64 x i8> addrspace(2)* %in) #0 {
  %load = load <64 x i8>, <64 x i8> addrspace(2)* %in
  %ext = sext <64 x i8> %load to <64 x i32>
  store <64 x i32> %ext, <64 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_zextload_i8_to_i64:
; GCN-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], 0{{$}}

; GCN-NOHSA-DAG: buffer_load_ubyte v[[LO:[0-9]+]],
; GCN-NOHSA: buffer_store_dwordx2 v{{\[}}[[LO]]:[[HI]]]

; GCN-HSA-DAG: flat_load_ubyte v[[LO:[0-9]+]],
; GCN-HSA: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[}}[[LO]]:[[HI]]]
define void @constant_zextload_i8_to_i64(i64 addrspace(1)* %out, i8 addrspace(2)* %in) #0 {
  %a = load i8, i8 addrspace(2)* %in
  %ext = zext i8 %a to i64
  store i64 %ext, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_sextload_i8_to_i64:
; GCN-NOHSA: buffer_load_sbyte v[[LO:[0-9]+]],
; GCN-HSA: flat_load_sbyte v[[LO:[0-9]+]],
; GCN: v_ashrrev_i32_e32 v[[HI:[0-9]+]], 31, v[[LO]]

; GCN-NOHSA: buffer_store_dwordx2 v{{\[}}[[LO]]:[[HI]]{{\]}}
; GCN-HSA: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[}}[[LO]]:[[HI]]{{\]}}
define void @constant_sextload_i8_to_i64(i64 addrspace(1)* %out, i8 addrspace(2)* %in) #0 {
  %a = load i8, i8 addrspace(2)* %in
  %ext = sext i8 %a to i64
  store i64 %ext, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_zextload_v1i8_to_v1i64:
define void @constant_zextload_v1i8_to_v1i64(<1 x i64> addrspace(1)* %out, <1 x i8> addrspace(2)* %in) #0 {
  %load = load <1 x i8>, <1 x i8> addrspace(2)* %in
  %ext = zext <1 x i8> %load to <1 x i64>
  store <1 x i64> %ext, <1 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_sextload_v1i8_to_v1i64:
define void @constant_sextload_v1i8_to_v1i64(<1 x i64> addrspace(1)* %out, <1 x i8> addrspace(2)* %in) #0 {
  %load = load <1 x i8>, <1 x i8> addrspace(2)* %in
  %ext = sext <1 x i8> %load to <1 x i64>
  store <1 x i64> %ext, <1 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_zextload_v2i8_to_v2i64:
define void @constant_zextload_v2i8_to_v2i64(<2 x i64> addrspace(1)* %out, <2 x i8> addrspace(2)* %in) #0 {
  %load = load <2 x i8>, <2 x i8> addrspace(2)* %in
  %ext = zext <2 x i8> %load to <2 x i64>
  store <2 x i64> %ext, <2 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_sextload_v2i8_to_v2i64:
define void @constant_sextload_v2i8_to_v2i64(<2 x i64> addrspace(1)* %out, <2 x i8> addrspace(2)* %in) #0 {
  %load = load <2 x i8>, <2 x i8> addrspace(2)* %in
  %ext = sext <2 x i8> %load to <2 x i64>
  store <2 x i64> %ext, <2 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_zextload_v4i8_to_v4i64:
define void @constant_zextload_v4i8_to_v4i64(<4 x i64> addrspace(1)* %out, <4 x i8> addrspace(2)* %in) #0 {
  %load = load <4 x i8>, <4 x i8> addrspace(2)* %in
  %ext = zext <4 x i8> %load to <4 x i64>
  store <4 x i64> %ext, <4 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_sextload_v4i8_to_v4i64:
define void @constant_sextload_v4i8_to_v4i64(<4 x i64> addrspace(1)* %out, <4 x i8> addrspace(2)* %in) #0 {
  %load = load <4 x i8>, <4 x i8> addrspace(2)* %in
  %ext = sext <4 x i8> %load to <4 x i64>
  store <4 x i64> %ext, <4 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_zextload_v8i8_to_v8i64:
define void @constant_zextload_v8i8_to_v8i64(<8 x i64> addrspace(1)* %out, <8 x i8> addrspace(2)* %in) #0 {
  %load = load <8 x i8>, <8 x i8> addrspace(2)* %in
  %ext = zext <8 x i8> %load to <8 x i64>
  store <8 x i64> %ext, <8 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_sextload_v8i8_to_v8i64:
define void @constant_sextload_v8i8_to_v8i64(<8 x i64> addrspace(1)* %out, <8 x i8> addrspace(2)* %in) #0 {
  %load = load <8 x i8>, <8 x i8> addrspace(2)* %in
  %ext = sext <8 x i8> %load to <8 x i64>
  store <8 x i64> %ext, <8 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_zextload_v16i8_to_v16i64:
define void @constant_zextload_v16i8_to_v16i64(<16 x i64> addrspace(1)* %out, <16 x i8> addrspace(2)* %in) #0 {
  %load = load <16 x i8>, <16 x i8> addrspace(2)* %in
  %ext = zext <16 x i8> %load to <16 x i64>
  store <16 x i64> %ext, <16 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_sextload_v16i8_to_v16i64:
define void @constant_sextload_v16i8_to_v16i64(<16 x i64> addrspace(1)* %out, <16 x i8> addrspace(2)* %in) #0 {
  %load = load <16 x i8>, <16 x i8> addrspace(2)* %in
  %ext = sext <16 x i8> %load to <16 x i64>
  store <16 x i64> %ext, <16 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_zextload_v32i8_to_v32i64:
define void @constant_zextload_v32i8_to_v32i64(<32 x i64> addrspace(1)* %out, <32 x i8> addrspace(2)* %in) #0 {
  %load = load <32 x i8>, <32 x i8> addrspace(2)* %in
  %ext = zext <32 x i8> %load to <32 x i64>
  store <32 x i64> %ext, <32 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_sextload_v32i8_to_v32i64:
define void @constant_sextload_v32i8_to_v32i64(<32 x i64> addrspace(1)* %out, <32 x i8> addrspace(2)* %in) #0 {
  %load = load <32 x i8>, <32 x i8> addrspace(2)* %in
  %ext = sext <32 x i8> %load to <32 x i64>
  store <32 x i64> %ext, <32 x i64> addrspace(1)* %out
  ret void
}

; XFUNC-LABEL: {{^}}constant_zextload_v64i8_to_v64i64:
; define void @constant_zextload_v64i8_to_v64i64(<64 x i64> addrspace(1)* %out, <64 x i8> addrspace(2)* %in) #0 {
;   %load = load <64 x i8>, <64 x i8> addrspace(2)* %in
;   %ext = zext <64 x i8> %load to <64 x i64>
;   store <64 x i64> %ext, <64 x i64> addrspace(1)* %out
;   ret void
; }

; XFUNC-LABEL: {{^}}constant_sextload_v64i8_to_v64i64:
; define void @constant_sextload_v64i8_to_v64i64(<64 x i64> addrspace(1)* %out, <64 x i8> addrspace(2)* %in) #0 {
;   %load = load <64 x i8>, <64 x i8> addrspace(2)* %in
;   %ext = sext <64 x i8> %load to <64 x i64>
;   store <64 x i64> %ext, <64 x i64> addrspace(1)* %out
;   ret void
; }

; FUNC-LABEL: {{^}}constant_zextload_i8_to_i16:
; GCN-NOHSA: buffer_load_ubyte v[[VAL:[0-9]+]],
; GCN-NOHSA: buffer_store_short v[[VAL]]

; GCN-HSA: flat_load_ubyte v[[VAL:[0-9]+]],
; GCN-HSA: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, v[[VAL]]
define void @constant_zextload_i8_to_i16(i16 addrspace(1)* %out, i8 addrspace(2)* %in) #0 {
  %a = load i8, i8 addrspace(2)* %in
  %ext = zext i8 %a to i16
  store i16 %ext, i16 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_sextload_i8_to_i16:
; GCN-NOHSA: buffer_load_sbyte v[[VAL:[0-9]+]],
; GCN-HSA: flat_load_sbyte v[[VAL:[0-9]+]],

; GCN-NOHSA: buffer_store_short v[[VAL]]
; GCN-HSA: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, v[[VAL]]
define void @constant_sextload_i8_to_i16(i16 addrspace(1)* %out, i8 addrspace(2)* %in) #0 {
  %a = load i8, i8 addrspace(2)* %in
  %ext = sext i8 %a to i16
  store i16 %ext, i16 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_zextload_v1i8_to_v1i16:
define void @constant_zextload_v1i8_to_v1i16(<1 x i16> addrspace(1)* %out, <1 x i8> addrspace(2)* %in) #0 {
  %load = load <1 x i8>, <1 x i8> addrspace(2)* %in
  %ext = zext <1 x i8> %load to <1 x i16>
  store <1 x i16> %ext, <1 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_sextload_v1i8_to_v1i16:
define void @constant_sextload_v1i8_to_v1i16(<1 x i16> addrspace(1)* %out, <1 x i8> addrspace(2)* %in) #0 {
  %load = load <1 x i8>, <1 x i8> addrspace(2)* %in
  %ext = sext <1 x i8> %load to <1 x i16>
  store <1 x i16> %ext, <1 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_zextload_v2i8_to_v2i16:
define void @constant_zextload_v2i8_to_v2i16(<2 x i16> addrspace(1)* %out, <2 x i8> addrspace(2)* %in) #0 {
  %load = load <2 x i8>, <2 x i8> addrspace(2)* %in
  %ext = zext <2 x i8> %load to <2 x i16>
  store <2 x i16> %ext, <2 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_sextload_v2i8_to_v2i16:
define void @constant_sextload_v2i8_to_v2i16(<2 x i16> addrspace(1)* %out, <2 x i8> addrspace(2)* %in) #0 {
  %load = load <2 x i8>, <2 x i8> addrspace(2)* %in
  %ext = sext <2 x i8> %load to <2 x i16>
  store <2 x i16> %ext, <2 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_zextload_v4i8_to_v4i16:
define void @constant_zextload_v4i8_to_v4i16(<4 x i16> addrspace(1)* %out, <4 x i8> addrspace(2)* %in) #0 {
  %load = load <4 x i8>, <4 x i8> addrspace(2)* %in
  %ext = zext <4 x i8> %load to <4 x i16>
  store <4 x i16> %ext, <4 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_sextload_v4i8_to_v4i16:
define void @constant_sextload_v4i8_to_v4i16(<4 x i16> addrspace(1)* %out, <4 x i8> addrspace(2)* %in) #0 {
  %load = load <4 x i8>, <4 x i8> addrspace(2)* %in
  %ext = sext <4 x i8> %load to <4 x i16>
  store <4 x i16> %ext, <4 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_zextload_v8i8_to_v8i16:
define void @constant_zextload_v8i8_to_v8i16(<8 x i16> addrspace(1)* %out, <8 x i8> addrspace(2)* %in) #0 {
  %load = load <8 x i8>, <8 x i8> addrspace(2)* %in
  %ext = zext <8 x i8> %load to <8 x i16>
  store <8 x i16> %ext, <8 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_sextload_v8i8_to_v8i16:
define void @constant_sextload_v8i8_to_v8i16(<8 x i16> addrspace(1)* %out, <8 x i8> addrspace(2)* %in) #0 {
  %load = load <8 x i8>, <8 x i8> addrspace(2)* %in
  %ext = sext <8 x i8> %load to <8 x i16>
  store <8 x i16> %ext, <8 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_zextload_v16i8_to_v16i16:
define void @constant_zextload_v16i8_to_v16i16(<16 x i16> addrspace(1)* %out, <16 x i8> addrspace(2)* %in) #0 {
  %load = load <16 x i8>, <16 x i8> addrspace(2)* %in
  %ext = zext <16 x i8> %load to <16 x i16>
  store <16 x i16> %ext, <16 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_sextload_v16i8_to_v16i16:
define void @constant_sextload_v16i8_to_v16i16(<16 x i16> addrspace(1)* %out, <16 x i8> addrspace(2)* %in) #0 {
  %load = load <16 x i8>, <16 x i8> addrspace(2)* %in
  %ext = sext <16 x i8> %load to <16 x i16>
  store <16 x i16> %ext, <16 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_zextload_v32i8_to_v32i16:
define void @constant_zextload_v32i8_to_v32i16(<32 x i16> addrspace(1)* %out, <32 x i8> addrspace(2)* %in) #0 {
  %load = load <32 x i8>, <32 x i8> addrspace(2)* %in
  %ext = zext <32 x i8> %load to <32 x i16>
  store <32 x i16> %ext, <32 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_sextload_v32i8_to_v32i16:
define void @constant_sextload_v32i8_to_v32i16(<32 x i16> addrspace(1)* %out, <32 x i8> addrspace(2)* %in) #0 {
  %load = load <32 x i8>, <32 x i8> addrspace(2)* %in
  %ext = sext <32 x i8> %load to <32 x i16>
  store <32 x i16> %ext, <32 x i16> addrspace(1)* %out
  ret void
}

; XFUNC-LABEL: {{^}}constant_zextload_v64i8_to_v64i16:
; define void @constant_zextload_v64i8_to_v64i16(<64 x i16> addrspace(1)* %out, <64 x i8> addrspace(2)* %in) #0 {
;   %load = load <64 x i8>, <64 x i8> addrspace(2)* %in
;   %ext = zext <64 x i8> %load to <64 x i16>
;   store <64 x i16> %ext, <64 x i16> addrspace(1)* %out
;   ret void
; }

; XFUNC-LABEL: {{^}}constant_sextload_v64i8_to_v64i16:
; define void @constant_sextload_v64i8_to_v64i16(<64 x i16> addrspace(1)* %out, <64 x i8> addrspace(2)* %in) #0 {
;   %load = load <64 x i8>, <64 x i8> addrspace(2)* %in
;   %ext = sext <64 x i8> %load to <64 x i16>
;   store <64 x i16> %ext, <64 x i16> addrspace(1)* %out
;   ret void
; }

attributes #0 = { nounwind }
