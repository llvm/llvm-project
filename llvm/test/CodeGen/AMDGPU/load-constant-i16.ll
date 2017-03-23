; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GCN-NOHSA,GCN-NOHSA-SI,FUNC %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN -check-prefix=GCN-HSA -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GCN-NOHSA,GCN-NOHSA-VI,FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}constant_load_i16:
; GCN-NOHSA: buffer_load_ushort v{{[0-9]+}}
; GCN-HSA: flat_load_ushort

; EG: VTX_READ_16 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0, #1
define amdgpu_kernel void @constant_load_i16(i16 addrspace(1)* %out, i16 addrspace(2)* %in) {
entry:
  %ld = load i16, i16 addrspace(2)* %in
  store i16 %ld, i16 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_load_v2i16:
; GCN: s_load_dword s

; EG: VTX_READ_32 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0, #1
define amdgpu_kernel void @constant_load_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(2)* %in) {
entry:
  %ld = load <2 x i16>, <2 x i16> addrspace(2)* %in
  store <2 x i16> %ld, <2 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_load_v3i16:
; GCN: s_load_dwordx2 s

; EG-DAG: VTX_READ_32 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0, #1
; EG-DAG: VTX_READ_16 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 4, #1
define amdgpu_kernel void @constant_load_v3i16(<3 x i16> addrspace(1)* %out, <3 x i16> addrspace(2)* %in) {
entry:
  %ld = load <3 x i16>, <3 x i16> addrspace(2)* %in
  store <3 x i16> %ld, <3 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_load_v4i16:
; GCN: s_load_dwordx2

; EG: VTX_READ_64 T{{[0-9]+}}.XY, T{{[0-9]+}}.X, 0, #1
define amdgpu_kernel void @constant_load_v4i16(<4 x i16> addrspace(1)* %out, <4 x i16> addrspace(2)* %in) {
entry:
  %ld = load <4 x i16>, <4 x i16> addrspace(2)* %in
  store <4 x i16> %ld, <4 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_load_v8i16:
; GCN: s_load_dwordx4

; EG: VTX_READ_128 T{{[0-9]+}}.XYZW, T{{[0-9]+}}.X, 0, #1
define amdgpu_kernel void @constant_load_v8i16(<8 x i16> addrspace(1)* %out, <8 x i16> addrspace(2)* %in) {
entry:
  %ld = load <8 x i16>, <8 x i16> addrspace(2)* %in
  store <8 x i16> %ld, <8 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_load_v16i16:
; GCN: s_load_dwordx8

; EG-DAG: VTX_READ_128 T{{[0-9]+}}.XYZW, T{{[0-9]+}}.X, 0, #1
; EG-DAG: VTX_READ_128 T{{[0-9]+}}.XYZW, T{{[0-9]+}}.X, 16, #1
define amdgpu_kernel void @constant_load_v16i16(<16 x i16> addrspace(1)* %out, <16 x i16> addrspace(2)* %in) {
entry:
  %ld = load <16 x i16>, <16 x i16> addrspace(2)* %in
  store <16 x i16> %ld, <16 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_zextload_i16_to_i32:
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_store_dword

; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_store_dword

; EG: VTX_READ_16 T{{[0-9]+\.X, T[0-9]+\.X}}, 0, #1
define amdgpu_kernel void @constant_zextload_i16_to_i32(i32 addrspace(1)* %out, i16 addrspace(2)* %in) #0 {
  %a = load i16, i16 addrspace(2)* %in
  %ext = zext i16 %a to i32
  store i32 %ext, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_sextload_i16_to_i32:
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_store_dword

; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_store_dword

; EG: VTX_READ_16 [[DST:T[0-9]\.[XYZW]]], [[DST]], 0, #1
; EG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, [[DST]], 0.0, literal
; EG: 16
define amdgpu_kernel void @constant_sextload_i16_to_i32(i32 addrspace(1)* %out, i16 addrspace(2)* %in) #0 {
  %a = load i16, i16 addrspace(2)* %in
  %ext = sext i16 %a to i32
  store i32 %ext, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_zextload_v1i16_to_v1i32:
; GCN-NOHSA: buffer_load_ushort
; GCN-HSA: flat_load_ushort

; EG: VTX_READ_16 T{{[0-9]+\.X, T[0-9]+\.X}}, 0, #1
define amdgpu_kernel void @constant_zextload_v1i16_to_v1i32(<1 x i32> addrspace(1)* %out, <1 x i16> addrspace(2)* %in) #0 {
  %load = load <1 x i16>, <1 x i16> addrspace(2)* %in
  %ext = zext <1 x i16> %load to <1 x i32>
  store <1 x i32> %ext, <1 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_sextload_v1i16_to_v1i32:
; GCN-NOHSA: buffer_load_sshort
; GCN-HSA: flat_load_sshort

; EG: VTX_READ_16 [[DST:T[0-9]\.[XYZW]]], [[DST]], 0, #1
; EG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, [[DST]], 0.0, literal
; EG: 16
define amdgpu_kernel void @constant_sextload_v1i16_to_v1i32(<1 x i32> addrspace(1)* %out, <1 x i16> addrspace(2)* %in) #0 {
  %load = load <1 x i16>, <1 x i16> addrspace(2)* %in
  %ext = sext <1 x i16> %load to <1 x i32>
  store <1 x i32> %ext, <1 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_zextload_v2i16_to_v2i32:
; GCN: s_load_dword s
; GCN-DAG: s_and_b32 s{{[0-9]+}}, s{{[0-9]+}}, 0xffff{{$}}
; GCN-DAG: s_lshr_b32 s{{[0-9]+}}, s{{[0-9]+}}, 16

; v2i16 is naturally 4 byte aligned
; EG: VTX_READ_32 [[DST:T[0-9]\.[XYZW]]], [[DST]], 0, #1
; EG: BFE_UINT {{[* ]*}}T{{[0-9].[XYZW]}}, [[DST]], literal
; EG: 16
; EG: 16
define amdgpu_kernel void @constant_zextload_v2i16_to_v2i32(<2 x i32> addrspace(1)* %out, <2 x i16> addrspace(2)* %in) #0 {
  %load = load <2 x i16>, <2 x i16> addrspace(2)* %in
  %ext = zext <2 x i16> %load to <2 x i32>
  store <2 x i32> %ext, <2 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_sextload_v2i16_to_v2i32:
; GCN: s_load_dword s
; GCN-DAG: s_ashr_i32
; GCN-DAG: s_sext_i32_i16

; v2i16 is naturally 4 byte aligned
; EG: MEM_RAT_CACHELESS STORE_RAW [[ST:T[0-9]]].XY, {{T[0-9].[XYZW]}},
; EG: VTX_READ_32 [[DST:T[0-9]\.[XYZW]]], [[DST]], 0, #1
; EG-DAG: BFE_INT {{[* ]*}}[[ST]].X, [[DST]], 0.0, literal
; TODO: We should use ASHR instead of LSHR + BFE
; EG-DAG: BFE_INT {{[* ]*}}[[ST]].Y, {{PV\.[XYZW]}}, 0.0, literal
; EG-DAG: 16
; EG-DAG: 16
define amdgpu_kernel void @constant_sextload_v2i16_to_v2i32(<2 x i32> addrspace(1)* %out, <2 x i16> addrspace(2)* %in) #0 {
  %load = load <2 x i16>, <2 x i16> addrspace(2)* %in
  %ext = sext <2 x i16> %load to <2 x i32>
  store <2 x i32> %ext, <2 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_zextload_v3i16_to_v3i32:
; GCN: s_load_dwordx2

; v3i16 is naturally 8 byte aligned
; EG-DAG: MEM_RAT_CACHELESS STORE_RAW [[ST_LO:T[0-9]]].XY, {{T[0-9].[XYZW]}},
; EG-DAG: MEM_RAT_CACHELESS STORE_RAW [[ST_HI:T[0-9]]].X, {{T[0-9].[XYZW]}},
; EG: CF_END
; EG-DAG: VTX_READ_32 [[DST_LO:T[0-9]\.[XYZW]]], {{T[0-9]\.[XYZW]}}, 0, #1
; EG-DAG: VTX_READ_16 [[DST_HI:T[0-9]\.[XYZW]]], {{T[0-9]\.[XYZW]}}, 4, #1
; TODO: This should use DST, but for some there are redundant MOVs
; EG-DAG: LSHR {{[* ]*}}[[ST_LO]].Y, {{T[0-9]\.[XYZW]}}, literal
; EG-DAG: 16
; EG-DAG: AND_INT {{[* ]*}}[[ST_LO]].X, {{T[0-9]\.[XYZW]}}, literal
; EG-DAG: AND_INT {{[* ]*}}[[ST_HI]].X, {{T[0-9]\.[XYZW]}}, literal
; EG-DAG: 65535
; EG-DAG: 65535
define amdgpu_kernel void @constant_zextload_v3i16_to_v3i32(<3 x i32> addrspace(1)* %out, <3 x i16> addrspace(2)* %in) {
entry:
  %ld = load <3 x i16>, <3 x i16> addrspace(2)* %in
  %ext = zext <3 x i16> %ld to <3 x i32>
  store <3 x i32> %ext, <3 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_sextload_v3i16_to_v3i32:
; GCN: s_load_dwordx2

; EG-DAG: MEM_RAT_CACHELESS STORE_RAW [[ST_LO:T[0-9]]].XY, {{T[0-9].[XYZW]}},
; EG-DAG: MEM_RAT_CACHELESS STORE_RAW [[ST_HI:T[0-9]]].X, {{T[0-9].[XYZW]}},
; v3i16 is naturally 8 byte aligned
; EG-DAG: VTX_READ_32 [[DST_HI:T[0-9]\.[XYZW]]], [[PTR:T[0-9]\.[XYZW]]], 0, #1
; EG-DAG: VTX_READ_16 [[DST_LO:T[0-9]\.[XYZW]]], {{T[0-9]\.[XYZW]}}, 4, #1
; EG-DAG: ASHR {{[* ]*}}[[ST_LO]].Y, {{T[0-9]\.[XYZW]}}, literal
; EG-DAG: BFE_INT {{[* ]*}}[[ST_LO]].X, {{T[0-9]\.[XYZW]}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}[[ST_HI]].X, {{T[0-9]\.[XYZW]}}, 0.0, literal
; EG-DAG: 16
; EG-DAG: 16
define amdgpu_kernel void @constant_sextload_v3i16_to_v3i32(<3 x i32> addrspace(1)* %out, <3 x i16> addrspace(2)* %in) {
entry:
  %ld = load <3 x i16>, <3 x i16> addrspace(2)* %in
  %ext = sext <3 x i16> %ld to <3 x i32>
  store <3 x i32> %ext, <3 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_zextload_v4i16_to_v4i32:
; GCN: s_load_dwordx2
; GCN-DAG: s_and_b32
; GCN-DAG: s_lshr_b32

; v4i16 is naturally 8 byte aligned
; EG: MEM_RAT_CACHELESS STORE_RAW [[ST:T[0-9]]].XYZW, {{T[0-9].[XYZW]}}
; EG: VTX_READ_64 [[LD:T[0-9]]].XY, {{T[0-9].[XYZW]}}, 0, #1
; TODO: This should use LD, but for some there are redundant MOVs
; EG-DAG: BFE_UINT {{[* ]*}}[[ST]].Y, {{.*\.[XYZW]}}, literal
; EG-DAG: BFE_UINT {{[* ]*}}[[ST]].W, {{.*\.[XYZW]}}, literal
; EG-DAG: 16
; EG-DAG: 16
; EG-DAG: AND_INT {{[* ]*}}[[ST]].X, {{T[0-9]\.[XYZW]}}, literal
; EG-DAG: AND_INT {{[* ]*}}[[ST]].Z, {{T[0-9]\.[XYZW]}}, literal
; EG-DAG: 65535
; EG-DAG: 65535
define amdgpu_kernel void @constant_zextload_v4i16_to_v4i32(<4 x i32> addrspace(1)* %out, <4 x i16> addrspace(2)* %in) #0 {
  %load = load <4 x i16>, <4 x i16> addrspace(2)* %in
  %ext = zext <4 x i16> %load to <4 x i32>
  store <4 x i32> %ext, <4 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_sextload_v4i16_to_v4i32:
; GCN: s_load_dwordx2
; GCN-DAG: s_ashr_i32
; GCN-DAG: s_sext_i32_i16

; v4i16 is naturally 8 byte aligned
; EG: MEM_RAT_CACHELESS STORE_RAW [[ST:T[0-9]]].XYZW, {{T[0-9]\.[XYZW]}},
; EG: VTX_READ_64 [[DST:T[0-9]]].XY, {{T[0-9].[XYZW]}}, 0, #1
; TODO: This should use LD, but for some there are redundant MOVs
; EG-DAG: BFE_INT {{[* ]*}}[[ST]].X, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}[[ST]].Z, {{.*}}, 0.0, literal
; TODO: We should use ASHR instead of LSHR + BFE
; EG-DAG: BFE_INT {{[* ]*}}[[ST]].Y, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}[[ST]].W, {{.*}}, 0.0, literal
; EG-DAG: 16
; EG-DAG: 16
; EG-DAG: 16
; EG-DAG: 16
define amdgpu_kernel void @constant_sextload_v4i16_to_v4i32(<4 x i32> addrspace(1)* %out, <4 x i16> addrspace(2)* %in) #0 {
  %load = load <4 x i16>, <4 x i16> addrspace(2)* %in
  %ext = sext <4 x i16> %load to <4 x i32>
  store <4 x i32> %ext, <4 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_zextload_v8i16_to_v8i32:
; GCN: s_load_dwordx4
; GCN-DAG: s_and_b32
; GCN-DAG: s_lshr_b32

; v8i16 is naturally 16 byte aligned
; EG: MEM_RAT_CACHELESS STORE_RAW [[ST_HI:T[0-9]]].XYZW, {{T[0-9]+.[XYZW]}},
; EG: MEM_RAT_CACHELESS STORE_RAW [[ST_LO:T[0-9]]].XYZW, {{T[0-9]+.[XYZW]}},
; EG: VTX_READ_128 [[DST:T[0-9]]].XYZW, {{T[0-9].[XYZW]}}, 0, #1
; TODO: These should use LSHR instead of BFE_UINT
; TODO: This should use DST, but for some there are redundant MOVs
; EG-DAG: BFE_UINT {{[* ]*}}[[ST_LO]].Y, {{.*}}, literal
; EG-DAG: BFE_UINT {{[* ]*}}[[ST_LO]].W, {{.*}}, literal
; EG-DAG: BFE_UINT {{[* ]*}}[[ST_HI]].Y, {{.*}}, literal
; EG-DAG: BFE_UINT {{[* ]*}}[[ST_HI]].W, {{.*}}, literal
; EG-DAG: AND_INT {{[* ]*}}[[ST_LO]].X, {{.*}}, literal
; EG-DAG: AND_INT {{[* ]*}}[[ST_LO]].Z, {{.*}}, literal
; EG-DAG: AND_INT {{[* ]*}}[[ST_HI]].X, {{.*}}, literal
; EG-DAG: AND_INT {{[* ]*}}[[ST_HI]].Z, {{.*}}, literal
; EG-DAG: 16
; EG-DAG: 16
; EG-DAG: 16
; EG-DAG: 16
; EG-DAG: 65535
; EG-DAG: 65535
; EG-DAG: 65535
; EG-DAG: 65535
define amdgpu_kernel void @constant_zextload_v8i16_to_v8i32(<8 x i32> addrspace(1)* %out, <8 x i16> addrspace(2)* %in) #0 {
  %load = load <8 x i16>, <8 x i16> addrspace(2)* %in
  %ext = zext <8 x i16> %load to <8 x i32>
  store <8 x i32> %ext, <8 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_sextload_v8i16_to_v8i32:
; GCN: s_load_dwordx4
; GCN-DAG: s_ashr_i32
; GCN-DAG: s_sext_i32_i16

; v8i16 is naturally 16 byte aligned
; EG: MEM_RAT_CACHELESS STORE_RAW [[ST_HI:T[0-9]]].XYZW, {{T[0-9]+.[XYZW]}},
; EG: MEM_RAT_CACHELESS STORE_RAW [[ST_LO:T[0-9]]].XYZW, {{T[0-9]+.[XYZW]}},
; EG: VTX_READ_128 [[DST:T[0-9]]].XYZW, {{T[0-9].[XYZW]}}, 0, #1
; TODO: 4 of these should use ASHR instead of LSHR + BFE_INT
; TODO: This should use DST, but for some there are redundant MOVs
; EG-DAG: BFE_INT {{[* ]*}}[[ST_LO]].Y, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}[[ST_LO]].W, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}[[ST_HI]].Y, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}[[ST_HI]].W, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}[[ST_LO]].X, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}[[ST_LO]].Z, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}[[ST_HI]].X, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}[[ST_HI]].Z, {{.*}}, 0.0, literal
; EG-DAG: 16
; EG-DAG: 16
; EG-DAG: 16
; EG-DAG: 16
; EG-DAG: 16
; EG-DAG: 16
; EG-DAG: 16
; EG-DAG: 16
define amdgpu_kernel void @constant_sextload_v8i16_to_v8i32(<8 x i32> addrspace(1)* %out, <8 x i16> addrspace(2)* %in) #0 {
  %load = load <8 x i16>, <8 x i16> addrspace(2)* %in
  %ext = sext <8 x i16> %load to <8 x i32>
  store <8 x i32> %ext, <8 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_zextload_v16i16_to_v16i32:
; GCN: s_load_dwordx8
; GCN-DAG: s_and_b32
; GCN-DAG: s_lshr_b32

; v16i16 is naturally 32 byte aligned
; EG-DAG: VTX_READ_128 [[DST_HI:T[0-9]+\.XYZW]], {{T[0-9]+.[XYZW]}}, 0, #1
; EG-DAG: VTX_READ_128 [[DST_LO:T[0-9]+\.XYZW]], {{T[0-9]+.[XYZW]}}, 16, #1
define amdgpu_kernel void @constant_zextload_v16i16_to_v16i32(<16 x i32> addrspace(1)* %out, <16 x i16> addrspace(2)* %in) #0 {
  %load = load <16 x i16>, <16 x i16> addrspace(2)* %in
  %ext = zext <16 x i16> %load to <16 x i32>
  store <16 x i32> %ext, <16 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_sextload_v16i16_to_v16i32:
; GCN: s_load_dwordx8
; GCN-DAG: s_ashr_i32
; GCN-DAG: s_sext_i32_i16

; v16i16 is naturally 32 byte aligned
; EG-DAG: VTX_READ_128 [[DST_HI:T[0-9]+\.XYZW]], {{T[0-9]+\.[XYZW]}}, 0, #1
; EG-DAG: VTX_READ_128 [[DST_LO:T[0-9]+\.XYZW]], {{T[0-9]+\.[XYZW]}}, 16, #1
define amdgpu_kernel void @constant_sextload_v16i16_to_v16i32(<16 x i32> addrspace(1)* %out, <16 x i16> addrspace(2)* %in) #0 {
  %load = load <16 x i16>, <16 x i16> addrspace(2)* %in
  %ext = sext <16 x i16> %load to <16 x i32>
  store <16 x i32> %ext, <16 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_zextload_v32i16_to_v32i32:
; GCN-DAG: s_load_dwordx16
; GCN-DAG: s_mov_b32 [[K:s[0-9]+]], 0xffff{{$}}
; GCN-DAG: s_lshr_b32 s{{[0-9]+}}, s{{[0-9]+}}, 16
; GCN-DAG: s_and_b32 s{{[0-9]+}}, s{{[0-9]+}}, [[K]]

; EG-DAG: VTX_READ_128 {{T[0-9]+\.XYZW}}, {{T[0-9]+\.[XYZW]}}, 0, #1
; EG-DAG: VTX_READ_128 {{T[0-9]+\.XYZW}}, {{T[0-9]+\.[XYZW]}}, 16, #1
; EG-DAG: VTX_READ_128 {{T[0-9]+\.XYZW}}, {{T[0-9]+\.[XYZW]}}, 32, #1
; EG-DAG: VTX_READ_128 {{T[0-9]+\.XYZW}}, {{T[0-9]+\.[XYZW]}}, 48, #1
define amdgpu_kernel void @constant_zextload_v32i16_to_v32i32(<32 x i32> addrspace(1)* %out, <32 x i16> addrspace(2)* %in) #0 {
  %load = load <32 x i16>, <32 x i16> addrspace(2)* %in
  %ext = zext <32 x i16> %load to <32 x i32>
  store <32 x i32> %ext, <32 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_sextload_v32i16_to_v32i32:
; GCN: s_load_dwordx16
; GCN-DAG: s_ashr_i32
; GCN-DAG: s_sext_i32_i16

; EG-DAG: VTX_READ_128 {{T[0-9]+\.XYZW}}, {{T[0-9]+\.[XYZW]}}, 0, #1
; EG-DAG: VTX_READ_128 {{T[0-9]+\.XYZW}}, {{T[0-9]+\.[XYZW]}}, 16, #1
; EG-DAG: VTX_READ_128 {{T[0-9]+\.XYZW}}, {{T[0-9]+\.[XYZW]}}, 32, #1
; EG-DAG: VTX_READ_128 {{T[0-9]+\.XYZW}}, {{T[0-9]+\.[XYZW]}}, 48, #1
define amdgpu_kernel void @constant_sextload_v32i16_to_v32i32(<32 x i32> addrspace(1)* %out, <32 x i16> addrspace(2)* %in) #0 {
  %load = load <32 x i16>, <32 x i16> addrspace(2)* %in
  %ext = sext <32 x i16> %load to <32 x i32>
  store <32 x i32> %ext, <32 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_zextload_v64i16_to_v64i32:
; GCN: s_load_dwordx16
; GCN: s_load_dwordx16

; EG-DAG: VTX_READ_128 {{T[0-9]+\.XYZW}}, {{T[0-9]+\.[XYZW]}}, 0, #1
; EG-DAG: VTX_READ_128 {{T[0-9]+\.XYZW}}, {{T[0-9]+\.[XYZW]}}, 16, #1
; EG-DAG: VTX_READ_128 {{T[0-9]+\.XYZW}}, {{T[0-9]+\.[XYZW]}}, 32, #1
; EG-DAG: VTX_READ_128 {{T[0-9]+\.XYZW}}, {{T[0-9]+\.[XYZW]}}, 48, #1
; EG-DAG: VTX_READ_128 {{T[0-9]+\.XYZW}}, {{T[0-9]+\.[XYZW]}}, 64, #1
; EG-DAG: VTX_READ_128 {{T[0-9]+\.XYZW}}, {{T[0-9]+\.[XYZW]}}, 80, #1
; EG-DAG: VTX_READ_128 {{T[0-9]+\.XYZW}}, {{T[0-9]+\.[XYZW]}}, 96, #1
; EG-DAG: VTX_READ_128 {{T[0-9]+\.XYZW}}, {{T[0-9]+\.[XYZW]}}, 112, #1
define amdgpu_kernel void @constant_zextload_v64i16_to_v64i32(<64 x i32> addrspace(1)* %out, <64 x i16> addrspace(2)* %in) #0 {
  %load = load <64 x i16>, <64 x i16> addrspace(2)* %in
  %ext = zext <64 x i16> %load to <64 x i32>
  store <64 x i32> %ext, <64 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_sextload_v64i16_to_v64i32:

; EG-DAG: VTX_READ_128 {{T[0-9]+\.XYZW}}, {{T[0-9]+\.[XYZW]}}, 0, #1
; EG-DAG: VTX_READ_128 {{T[0-9]+\.XYZW}}, {{T[0-9]+\.[XYZW]}}, 16, #1
; EG-DAG: VTX_READ_128 {{T[0-9]+\.XYZW}}, {{T[0-9]+\.[XYZW]}}, 32, #1
; EG-DAG: VTX_READ_128 {{T[0-9]+\.XYZW}}, {{T[0-9]+\.[XYZW]}}, 48, #1
; EG-DAG: VTX_READ_128 {{T[0-9]+\.XYZW}}, {{T[0-9]+\.[XYZW]}}, 64, #1
; EG-DAG: VTX_READ_128 {{T[0-9]+\.XYZW}}, {{T[0-9]+\.[XYZW]}}, 80, #1
; EG-DAG: VTX_READ_128 {{T[0-9]+\.XYZW}}, {{T[0-9]+\.[XYZW]}}, 96, #1
; EG-DAG: VTX_READ_128 {{T[0-9]+\.XYZW}}, {{T[0-9]+\.[XYZW]}}, 112, #1
define amdgpu_kernel void @constant_sextload_v64i16_to_v64i32(<64 x i32> addrspace(1)* %out, <64 x i16> addrspace(2)* %in) #0 {
  %load = load <64 x i16>, <64 x i16> addrspace(2)* %in
  %ext = sext <64 x i16> %load to <64 x i32>
  store <64 x i32> %ext, <64 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_zextload_i16_to_i64:
; GCN-NOHSA-DAG: buffer_load_ushort v[[LO:[0-9]+]],
; GCN-HSA-DAG: flat_load_ushort v[[LO:[0-9]+]],
; GCN-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], 0{{$}}

; GCN-NOHSA: buffer_store_dwordx2 v{{\[}}[[LO]]:[[HI]]]
; GCN-HSA: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[}}[[LO]]:[[HI]]{{\]}}

; EG: VTX_READ_16 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0, #1
; EG: MOV {{.*}}, 0.0
define amdgpu_kernel void @constant_zextload_i16_to_i64(i64 addrspace(1)* %out, i16 addrspace(2)* %in) #0 {
  %a = load i16, i16 addrspace(2)* %in
  %ext = zext i16 %a to i64
  store i64 %ext, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_sextload_i16_to_i64:
; FIXME: Need to optimize this sequence to avoid extra bfe:
;  t28: i32,ch = load<LD2[%in(addrspace=1)], anyext from i16> t12, t27, undef:i64
;          t31: i64 = any_extend t28 
;        t33: i64 = sign_extend_inreg t31, ValueType:ch:i16

; GCN-NOHSA-SI-DAG: buffer_load_sshort v[[LO:[0-9]+]],
; GCN-HSA-DAG: flat_load_sshort v[[LO:[0-9]+]],
; GCN-NOHSA-VI-DAG: buffer_load_ushort v[[ULO:[0-9]+]],
; GCN-NOHSA-VI-DAG: v_bfe_i32 v[[LO:[0-9]+]], v[[ULO]], 0, 16
; GCN-DAG: v_ashrrev_i32_e32 v[[HI:[0-9]+]], 31, v[[LO]]

; GCN-NOHSA: buffer_store_dwordx2 v{{\[}}[[LO]]:[[HI]]]
; GCN-HSA: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[}}[[LO]]:[[HI]]{{\]}}

; EG: VTX_READ_16 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0, #1
; EG: ASHR {{\**}} {{T[0-9]\.[XYZW]}}, {{.*}}, literal
; TODO: These could be expanded earlier using ASHR 15
; EG: 31
define amdgpu_kernel void @constant_sextload_i16_to_i64(i64 addrspace(1)* %out, i16 addrspace(2)* %in) #0 {
  %a = load i16, i16 addrspace(2)* %in
  %ext = sext i16 %a to i64
  store i64 %ext, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_zextload_v1i16_to_v1i64:

; EG: VTX_READ_16 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0, #1
; EG: MOV {{.*}}, 0.0
define amdgpu_kernel void @constant_zextload_v1i16_to_v1i64(<1 x i64> addrspace(1)* %out, <1 x i16> addrspace(2)* %in) #0 {
  %load = load <1 x i16>, <1 x i16> addrspace(2)* %in
  %ext = zext <1 x i16> %load to <1 x i64>
  store <1 x i64> %ext, <1 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_sextload_v1i16_to_v1i64:

; EG: VTX_READ_16 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0, #1
; EG: ASHR {{\**}} {{T[0-9]\.[XYZW]}}, {{.*}}, literal
; TODO: These could be expanded earlier using ASHR 15
; EG: 31
define amdgpu_kernel void @constant_sextload_v1i16_to_v1i64(<1 x i64> addrspace(1)* %out, <1 x i16> addrspace(2)* %in) #0 {
  %load = load <1 x i16>, <1 x i16> addrspace(2)* %in
  %ext = sext <1 x i16> %load to <1 x i64>
  store <1 x i64> %ext, <1 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_zextload_v2i16_to_v2i64:

; EG: VTX_READ_32 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0, #1
define amdgpu_kernel void @constant_zextload_v2i16_to_v2i64(<2 x i64> addrspace(1)* %out, <2 x i16> addrspace(2)* %in) #0 {
  %load = load <2 x i16>, <2 x i16> addrspace(2)* %in
  %ext = zext <2 x i16> %load to <2 x i64>
  store <2 x i64> %ext, <2 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_sextload_v2i16_to_v2i64:

; EG: VTX_READ_32 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0, #1
define amdgpu_kernel void @constant_sextload_v2i16_to_v2i64(<2 x i64> addrspace(1)* %out, <2 x i16> addrspace(2)* %in) #0 {
  %load = load <2 x i16>, <2 x i16> addrspace(2)* %in
  %ext = sext <2 x i16> %load to <2 x i64>
  store <2 x i64> %ext, <2 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_zextload_v4i16_to_v4i64:

; EG: VTX_READ_64 T{{[0-9]+}}.XY, T{{[0-9]+}}.X, 0, #1
define amdgpu_kernel void @constant_zextload_v4i16_to_v4i64(<4 x i64> addrspace(1)* %out, <4 x i16> addrspace(2)* %in) #0 {
  %load = load <4 x i16>, <4 x i16> addrspace(2)* %in
  %ext = zext <4 x i16> %load to <4 x i64>
  store <4 x i64> %ext, <4 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_sextload_v4i16_to_v4i64:

; EG: VTX_READ_64 T{{[0-9]+}}.XY, T{{[0-9]+}}.X, 0, #1
define amdgpu_kernel void @constant_sextload_v4i16_to_v4i64(<4 x i64> addrspace(1)* %out, <4 x i16> addrspace(2)* %in) #0 {
  %load = load <4 x i16>, <4 x i16> addrspace(2)* %in
  %ext = sext <4 x i16> %load to <4 x i64>
  store <4 x i64> %ext, <4 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_zextload_v8i16_to_v8i64:

; EG: VTX_READ_128 T{{[0-9]+}}.XYZW, T{{[0-9]+}}.X, 0, #1
define amdgpu_kernel void @constant_zextload_v8i16_to_v8i64(<8 x i64> addrspace(1)* %out, <8 x i16> addrspace(2)* %in) #0 {
  %load = load <8 x i16>, <8 x i16> addrspace(2)* %in
  %ext = zext <8 x i16> %load to <8 x i64>
  store <8 x i64> %ext, <8 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_sextload_v8i16_to_v8i64:

; EG: VTX_READ_128 T{{[0-9]+}}.XYZW, T{{[0-9]+}}.X, 0, #1
define amdgpu_kernel void @constant_sextload_v8i16_to_v8i64(<8 x i64> addrspace(1)* %out, <8 x i16> addrspace(2)* %in) #0 {
  %load = load <8 x i16>, <8 x i16> addrspace(2)* %in
  %ext = sext <8 x i16> %load to <8 x i64>
  store <8 x i64> %ext, <8 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_zextload_v16i16_to_v16i64:

; EG-DAG: VTX_READ_128 T{{[0-9]+}}.XYZW, T{{[0-9]+}}.X, 0, #1
; EG-DAG: VTX_READ_128 T{{[0-9]+}}.XYZW, T{{[0-9]+}}.X, 16, #1
define amdgpu_kernel void @constant_zextload_v16i16_to_v16i64(<16 x i64> addrspace(1)* %out, <16 x i16> addrspace(2)* %in) #0 {
  %load = load <16 x i16>, <16 x i16> addrspace(2)* %in
  %ext = zext <16 x i16> %load to <16 x i64>
  store <16 x i64> %ext, <16 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_sextload_v16i16_to_v16i64:

; EG-DAG: VTX_READ_128 T{{[0-9]+}}.XYZW, T{{[0-9]+}}.X, 0, #1
; EG-DAG: VTX_READ_128 T{{[0-9]+}}.XYZW, T{{[0-9]+}}.X, 16, #1
define amdgpu_kernel void @constant_sextload_v16i16_to_v16i64(<16 x i64> addrspace(1)* %out, <16 x i16> addrspace(2)* %in) #0 {
  %load = load <16 x i16>, <16 x i16> addrspace(2)* %in
  %ext = sext <16 x i16> %load to <16 x i64>
  store <16 x i64> %ext, <16 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_zextload_v32i16_to_v32i64:

; EG-DAG: VTX_READ_128 T{{[0-9]+}}.XYZW, T{{[0-9]+}}.X, 0, #1
; EG-DAG: VTX_READ_128 T{{[0-9]+}}.XYZW, T{{[0-9]+}}.X, 16, #1
; EG-DAG: VTX_READ_128 T{{[0-9]+}}.XYZW, T{{[0-9]+}}.X, 32, #1
; EG-DAG: VTX_READ_128 T{{[0-9]+}}.XYZW, T{{[0-9]+}}.X, 48, #1
define amdgpu_kernel void @constant_zextload_v32i16_to_v32i64(<32 x i64> addrspace(1)* %out, <32 x i16> addrspace(2)* %in) #0 {
  %load = load <32 x i16>, <32 x i16> addrspace(2)* %in
  %ext = zext <32 x i16> %load to <32 x i64>
  store <32 x i64> %ext, <32 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_sextload_v32i16_to_v32i64:

; EG-DAG: VTX_READ_128 T{{[0-9]+}}.XYZW, T{{[0-9]+}}.X, 0, #1
; EG-DAG: VTX_READ_128 T{{[0-9]+}}.XYZW, T{{[0-9]+}}.X, 16, #1
; EG-DAG: VTX_READ_128 T{{[0-9]+}}.XYZW, T{{[0-9]+}}.X, 32, #1
; EG-DAG: VTX_READ_128 T{{[0-9]+}}.XYZW, T{{[0-9]+}}.X, 48, #1
define amdgpu_kernel void @constant_sextload_v32i16_to_v32i64(<32 x i64> addrspace(1)* %out, <32 x i16> addrspace(2)* %in) #0 {
  %load = load <32 x i16>, <32 x i16> addrspace(2)* %in
  %ext = sext <32 x i16> %load to <32 x i64>
  store <32 x i64> %ext, <32 x i64> addrspace(1)* %out
  ret void
}

; These trigger undefined register machine verifier errors

; ; XFUNC-LABEL: {{^}}constant_zextload_v64i16_to_v64i64:
; define amdgpu_kernel void @constant_zextload_v64i16_to_v64i64(<64 x i64> addrspace(1)* %out, <64 x i16> addrspace(2)* %in) #0 {
;   %load = load <64 x i16>, <64 x i16> addrspace(2)* %in
;   %ext = zext <64 x i16> %load to <64 x i64>
;   store <64 x i64> %ext, <64 x i64> addrspace(1)* %out
;   ret void
; }

; ; XFUNC-LABEL: {{^}}constant_sextload_v64i16_to_v64i64:
; define amdgpu_kernel void @constant_sextload_v64i16_to_v64i64(<64 x i64> addrspace(1)* %out, <64 x i16> addrspace(2)* %in) #0 {
;   %load = load <64 x i16>, <64 x i16> addrspace(2)* %in
;   %ext = sext <64 x i16> %load to <64 x i64>
;   store <64 x i64> %ext, <64 x i64> addrspace(1)* %out
;   ret void
; }

attributes #0 = { nounwind }
