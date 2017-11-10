; RUN: llc -march=amdgcn -mtriple=amdgcn---amdgiz -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SI,FUNC %s
; RUN: llc -march=amdgcn -mtriple=amdgcn---amdgiz -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,VI,FUNC %s
; RUN: llc -march=r600 -mtriple=r600---amdgiz -mcpu=redwood -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s


; FUNC-LABEL: {{^}}local_load_i8:
; GCN-NOT: s_wqm_b64
; GCN: s_mov_b32 m0
; GCN: ds_read_u8

; EG: LDS_UBYTE_READ_RET
define amdgpu_kernel void @local_load_i8(i8 addrspace(3)* %out, i8 addrspace(3)* %in) #0 {
entry:
  %ld = load i8, i8 addrspace(3)* %in
  store i8 %ld, i8 addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_load_v2i8:
; GCN-NOT: s_wqm_b64
; GCN: s_mov_b32 m0
; GCN: ds_read_u16

; EG: LDS_USHORT_READ_RET
define amdgpu_kernel void @local_load_v2i8(<2 x i8> addrspace(3)* %out, <2 x i8> addrspace(3)* %in) #0 {
entry:
  %ld = load <2 x i8>, <2 x i8> addrspace(3)* %in
  store <2 x i8> %ld, <2 x i8> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_load_v3i8:
; GCN: ds_read_b32

; EG: DS_READ_RET
define amdgpu_kernel void @local_load_v3i8(<3 x i8> addrspace(3)* %out, <3 x i8> addrspace(3)* %in) #0 {
entry:
  %ld = load <3 x i8>, <3 x i8> addrspace(3)* %in
  store <3 x i8> %ld, <3 x i8> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_load_v4i8:
; GCN: ds_read_b32

; EG: LDS_READ_RET
define amdgpu_kernel void @local_load_v4i8(<4 x i8> addrspace(3)* %out, <4 x i8> addrspace(3)* %in) #0 {
entry:
  %ld = load <4 x i8>, <4 x i8> addrspace(3)* %in
  store <4 x i8> %ld, <4 x i8> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_load_v8i8:
; GCN: ds_read_b64

; EG: LDS_READ_RET
; EG: LDS_READ_RET
define amdgpu_kernel void @local_load_v8i8(<8 x i8> addrspace(3)* %out, <8 x i8> addrspace(3)* %in) #0 {
entry:
  %ld = load <8 x i8>, <8 x i8> addrspace(3)* %in
  store <8 x i8> %ld, <8 x i8> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_load_v16i8:
; GCN: ds_read2_b64  v{{\[}}[[LO:[0-9]+]]:[[HI:[0-9]+]]{{\]}}, v{{[0-9]+}} offset1:1{{$}}
; GCN: ds_write2_b64 v{{[0-9]+}}, v{{\[}}[[LO]]:{{[0-9]+}}], v[{{[0-9]+}}:[[HI]]{{\]}} offset1:1{{$}}

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
define amdgpu_kernel void @local_load_v16i8(<16 x i8> addrspace(3)* %out, <16 x i8> addrspace(3)* %in) #0 {
entry:
  %ld = load <16 x i8>, <16 x i8> addrspace(3)* %in
  store <16 x i8> %ld, <16 x i8> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_i8_to_i32:
; GCN-NOT: s_wqm_b64
; GCN: s_mov_b32 m0
; GCN: ds_read_u8

; EG: LDS_UBYTE_READ_RET
define amdgpu_kernel void @local_zextload_i8_to_i32(i32 addrspace(3)* %out, i8 addrspace(3)* %in) #0 {
  %a = load i8, i8 addrspace(3)* %in
  %ext = zext i8 %a to i32
  store i32 %ext, i32 addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_i8_to_i32:
; GCN-NOT: s_wqm_b64
; GCN: s_mov_b32 m0
; GCN: ds_read_i8

; EG: LDS_UBYTE_READ_RET
; EG: BFE_INT
define amdgpu_kernel void @local_sextload_i8_to_i32(i32 addrspace(3)* %out, i8 addrspace(3)* %in) #0 {
  %ld = load i8, i8 addrspace(3)* %in
  %ext = sext i8 %ld to i32
  store i32 %ext, i32 addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v1i8_to_v1i32:

; EG: LDS_UBYTE_READ_RET
define amdgpu_kernel void @local_zextload_v1i8_to_v1i32(<1 x i32> addrspace(3)* %out, <1 x i8> addrspace(3)* %in) #0 {
  %load = load <1 x i8>, <1 x i8> addrspace(3)* %in
  %ext = zext <1 x i8> %load to <1 x i32>
  store <1 x i32> %ext, <1 x i32> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v1i8_to_v1i32:

; EG: LDS_UBYTE_READ_RET
; EG: BFE_INT
define amdgpu_kernel void @local_sextload_v1i8_to_v1i32(<1 x i32> addrspace(3)* %out, <1 x i8> addrspace(3)* %in) #0 {
  %load = load <1 x i8>, <1 x i8> addrspace(3)* %in
  %ext = sext <1 x i8> %load to <1 x i32>
  store <1 x i32> %ext, <1 x i32> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v2i8_to_v2i32:
; GCN: ds_read_u16

; EG: LDS_USHORT_READ_RET
define amdgpu_kernel void @local_zextload_v2i8_to_v2i32(<2 x i32> addrspace(3)* %out, <2 x i8> addrspace(3)* %in) #0 {
  %load = load <2 x i8>, <2 x i8> addrspace(3)* %in
  %ext = zext <2 x i8> %load to <2 x i32>
  store <2 x i32> %ext, <2 x i32> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v2i8_to_v2i32:
; GCN-NOT: s_wqm_b64
; GCN: s_mov_b32 m0
; GCN: ds_read_u16
; FIXME: Need to optimize this sequence to avoid extra shift on VI.
;         t23: i16 = srl t39, Constant:i32<8>
;          t31: i32 = any_extend t23
;        t33: i32 = sign_extend_inreg t31, ValueType:ch:i8

; SI-DAG: v_bfe_i32 v{{[0-9]+}}, v{{[0-9]+}}, 8, 8
; SI-DAG: v_bfe_i32 v{{[0-9]+}}, v{{[0-9]+}}, 0, 8

; VI-DAG: v_lshrrev_b16_e32 [[SHIFT:v[0-9]+]], 8, v{{[0-9]+}}
; VI-DAG: v_bfe_i32 v{{[0-9]+}}, v{{[0-9]+}}, 0, 8
; VI-DAG: v_bfe_i32 v{{[0-9]+}}, [[SHIFT]], 0, 8

; EG: LDS_USHORT_READ_RET
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
define amdgpu_kernel void @local_sextload_v2i8_to_v2i32(<2 x i32> addrspace(3)* %out, <2 x i8> addrspace(3)* %in) #0 {
  %load = load <2 x i8>, <2 x i8> addrspace(3)* %in
  %ext = sext <2 x i8> %load to <2 x i32>
  store <2 x i32> %ext, <2 x i32> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v3i8_to_v3i32:
; GCN: ds_read_b32

; SI-DAG: v_bfe_u32 v{{[0-9]+}}, v{{[0-9]+}}, 8, 8
; VI-DAG: v_lshrrev_b16_e32 v{{[0-9]+}}, 8, {{v[0-9]+}}
; GCN-DAG: v_bfe_u32 v{{[0-9]+}}, v{{[0-9]+}}, 16, 8
; GCN-DAG: v_and_b32_e32 v{{[0-9]+}}, 0xff,

; EG: LDS_READ_RET
define amdgpu_kernel void @local_zextload_v3i8_to_v3i32(<3 x i32> addrspace(3)* %out, <3 x i8> addrspace(3)* %in) #0 {
entry:
  %ld = load <3 x i8>, <3 x i8> addrspace(3)* %in
  %ext = zext <3 x i8> %ld to <3 x i32>
  store <3 x i32> %ext, <3 x i32> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v3i8_to_v3i32:
; GCN-NOT: s_wqm_b64
; GCN: s_mov_b32 m0
; GCN: ds_read_b32

; GCN-DAG: v_bfe_i32
; GCN-DAG: v_bfe_i32
; GCN-DAG: v_bfe_i32
; GCN-DAG: v_bfe_i32

; GCN-DAG: ds_write_b64
; GCN-DAG: ds_write_b32

; EG: LDS_READ_RET
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
define amdgpu_kernel void @local_sextload_v3i8_to_v3i32(<3 x i32> addrspace(3)* %out, <3 x i8> addrspace(3)* %in) #0 {
entry:
  %ld = load <3 x i8>, <3 x i8> addrspace(3)* %in
  %ext = sext <3 x i8> %ld to <3 x i32>
  store <3 x i32> %ext, <3 x i32> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v4i8_to_v4i32:
; GCN-NOT: s_wqm_b64
; GCN: s_mov_b32 m0
; GCN: ds_read_b32

; EG: LDS_READ_RET
; EG-DAG: BFE_UINT
; EG-DAG: BFE_UINT
; EG-DAG: BFE_UINT
define amdgpu_kernel void @local_zextload_v4i8_to_v4i32(<4 x i32> addrspace(3)* %out, <4 x i8> addrspace(3)* %in) #0 {
  %load = load <4 x i8>, <4 x i8> addrspace(3)* %in
  %ext = zext <4 x i8> %load to <4 x i32>
  store <4 x i32> %ext, <4 x i32> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v4i8_to_v4i32:
; GCN-NOT: s_wqm_b64
; GCN: s_mov_b32 m0
; GCN: ds_read_b32

; EG-DAG: LDS_READ_RET
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
define amdgpu_kernel void @local_sextload_v4i8_to_v4i32(<4 x i32> addrspace(3)* %out, <4 x i8> addrspace(3)* %in) #0 {
  %load = load <4 x i8>, <4 x i8> addrspace(3)* %in
  %ext = sext <4 x i8> %load to <4 x i32>
  store <4 x i32> %ext, <4 x i32> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v8i8_to_v8i32:

; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: BFE_UINT
; EG-DAG: BFE_UINT
; EG-DAG: BFE_UINT
; EG-DAG: BFE_UINT
; EG-DAG: BFE_UINT
; EG-DAG: BFE_UINT
define amdgpu_kernel void @local_zextload_v8i8_to_v8i32(<8 x i32> addrspace(3)* %out, <8 x i8> addrspace(3)* %in) #0 {
  %load = load <8 x i8>, <8 x i8> addrspace(3)* %in
  %ext = zext <8 x i8> %load to <8 x i32>
  store <8 x i32> %ext, <8 x i32> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v8i8_to_v8i32:

; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
define amdgpu_kernel void @local_sextload_v8i8_to_v8i32(<8 x i32> addrspace(3)* %out, <8 x i8> addrspace(3)* %in) #0 {
  %load = load <8 x i8>, <8 x i8> addrspace(3)* %in
  %ext = sext <8 x i8> %load to <8 x i32>
  store <8 x i32> %ext, <8 x i32> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v16i8_to_v16i32:

; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: BFE_UINT
; EG-DAG: BFE_UINT
; EG-DAG: BFE_UINT
; EG-DAG: BFE_UINT
; EG-DAG: BFE_UINT
; EG-DAG: BFE_UINT
; EG-DAG: BFE_UINT
; EG-DAG: BFE_UINT
; EG-DAG: BFE_UINT
; EG-DAG: BFE_UINT
; EG-DAG: BFE_UINT
; EG-DAG: BFE_UINT
define amdgpu_kernel void @local_zextload_v16i8_to_v16i32(<16 x i32> addrspace(3)* %out, <16 x i8> addrspace(3)* %in) #0 {
  %load = load <16 x i8>, <16 x i8> addrspace(3)* %in
  %ext = zext <16 x i8> %load to <16 x i32>
  store <16 x i32> %ext, <16 x i32> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v16i8_to_v16i32:

; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
define amdgpu_kernel void @local_sextload_v16i8_to_v16i32(<16 x i32> addrspace(3)* %out, <16 x i8> addrspace(3)* %in) #0 {
  %load = load <16 x i8>, <16 x i8> addrspace(3)* %in
  %ext = sext <16 x i8> %load to <16 x i32>
  store <16 x i32> %ext, <16 x i32> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v32i8_to_v32i32:

; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
define amdgpu_kernel void @local_zextload_v32i8_to_v32i32(<32 x i32> addrspace(3)* %out, <32 x i8> addrspace(3)* %in) #0 {
  %load = load <32 x i8>, <32 x i8> addrspace(3)* %in
  %ext = zext <32 x i8> %load to <32 x i32>
  store <32 x i32> %ext, <32 x i32> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v32i8_to_v32i32:

; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
define amdgpu_kernel void @local_sextload_v32i8_to_v32i32(<32 x i32> addrspace(3)* %out, <32 x i8> addrspace(3)* %in) #0 {
  %load = load <32 x i8>, <32 x i8> addrspace(3)* %in
  %ext = sext <32 x i8> %load to <32 x i32>
  store <32 x i32> %ext, <32 x i32> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v64i8_to_v64i32:

; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
define amdgpu_kernel void @local_zextload_v64i8_to_v64i32(<64 x i32> addrspace(3)* %out, <64 x i8> addrspace(3)* %in) #0 {
  %load = load <64 x i8>, <64 x i8> addrspace(3)* %in
  %ext = zext <64 x i8> %load to <64 x i32>
  store <64 x i32> %ext, <64 x i32> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v64i8_to_v64i32:

; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
; EG-DAG: LDS_READ_RET
define amdgpu_kernel void @local_sextload_v64i8_to_v64i32(<64 x i32> addrspace(3)* %out, <64 x i8> addrspace(3)* %in) #0 {
  %load = load <64 x i8>, <64 x i8> addrspace(3)* %in
  %ext = sext <64 x i8> %load to <64 x i32>
  store <64 x i32> %ext, <64 x i32> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_i8_to_i64:
; GCN-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], 0{{$}}
; GCN-DAG: ds_read_u8 v[[LO:[0-9]+]],
; GCN: ds_write_b64 v{{[0-9]+}}, v{{\[}}[[LO]]:[[HI]]]

; EG: LDS_UBYTE_READ_RET
; EG: MOV {{.*}}, literal
; EG: 0.0
define amdgpu_kernel void @local_zextload_i8_to_i64(i64 addrspace(3)* %out, i8 addrspace(3)* %in) #0 {
  %a = load i8, i8 addrspace(3)* %in
  %ext = zext i8 %a to i64
  store i64 %ext, i64 addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_i8_to_i64:
; GCN: ds_read_i8 v[[LO:[0-9]+]],
; GCN: v_ashrrev_i32_e32 v[[HI:[0-9]+]], 31, v[[LO]]

; GCN: ds_write_b64 v{{[0-9]+}}, v{{\[}}[[LO]]:[[HI]]{{\]}}

; EG: LDS_UBYTE_READ_RET
; EG: ASHR
; TODO: why not 7?
; EG: 31
define amdgpu_kernel void @local_sextload_i8_to_i64(i64 addrspace(3)* %out, i8 addrspace(3)* %in) #0 {
  %a = load i8, i8 addrspace(3)* %in
  %ext = sext i8 %a to i64
  store i64 %ext, i64 addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v1i8_to_v1i64:

; EG: LDS_UBYTE_READ_RET
; EG: MOV {{.*}}, literal
; TODO: merge?
; EG: 0.0
define amdgpu_kernel void @local_zextload_v1i8_to_v1i64(<1 x i64> addrspace(3)* %out, <1 x i8> addrspace(3)* %in) #0 {
  %load = load <1 x i8>, <1 x i8> addrspace(3)* %in
  %ext = zext <1 x i8> %load to <1 x i64>
  store <1 x i64> %ext, <1 x i64> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v1i8_to_v1i64:

; EG: LDS_UBYTE_READ_RET
; EG: ASHR
; TODO: why not 7?
; EG: 31
define amdgpu_kernel void @local_sextload_v1i8_to_v1i64(<1 x i64> addrspace(3)* %out, <1 x i8> addrspace(3)* %in) #0 {
  %load = load <1 x i8>, <1 x i8> addrspace(3)* %in
  %ext = sext <1 x i8> %load to <1 x i64>
  store <1 x i64> %ext, <1 x i64> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v2i8_to_v2i64:

; EG: LDS_USHORT_READ_RET
define amdgpu_kernel void @local_zextload_v2i8_to_v2i64(<2 x i64> addrspace(3)* %out, <2 x i8> addrspace(3)* %in) #0 {
  %load = load <2 x i8>, <2 x i8> addrspace(3)* %in
  %ext = zext <2 x i8> %load to <2 x i64>
  store <2 x i64> %ext, <2 x i64> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v2i8_to_v2i64:

; EG: LDS_USHORT_READ_RET
; EG: BFE_INT
; EG: BFE_INT
define amdgpu_kernel void @local_sextload_v2i8_to_v2i64(<2 x i64> addrspace(3)* %out, <2 x i8> addrspace(3)* %in) #0 {
  %load = load <2 x i8>, <2 x i8> addrspace(3)* %in
  %ext = sext <2 x i8> %load to <2 x i64>
  store <2 x i64> %ext, <2 x i64> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v4i8_to_v4i64:

; EG: LDS_READ_RET
define amdgpu_kernel void @local_zextload_v4i8_to_v4i64(<4 x i64> addrspace(3)* %out, <4 x i8> addrspace(3)* %in) #0 {
  %load = load <4 x i8>, <4 x i8> addrspace(3)* %in
  %ext = zext <4 x i8> %load to <4 x i64>
  store <4 x i64> %ext, <4 x i64> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v4i8_to_v4i64:

; EG: LDS_READ_RET
define amdgpu_kernel void @local_sextload_v4i8_to_v4i64(<4 x i64> addrspace(3)* %out, <4 x i8> addrspace(3)* %in) #0 {
  %load = load <4 x i8>, <4 x i8> addrspace(3)* %in
  %ext = sext <4 x i8> %load to <4 x i64>
  store <4 x i64> %ext, <4 x i64> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v8i8_to_v8i64:

; EG: LDS_READ_RET
; EG: LDS_READ_RET
define amdgpu_kernel void @local_zextload_v8i8_to_v8i64(<8 x i64> addrspace(3)* %out, <8 x i8> addrspace(3)* %in) #0 {
  %load = load <8 x i8>, <8 x i8> addrspace(3)* %in
  %ext = zext <8 x i8> %load to <8 x i64>
  store <8 x i64> %ext, <8 x i64> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v8i8_to_v8i64:

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG-DAG: ASHR
; EG-DAG: ASHR
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
define amdgpu_kernel void @local_sextload_v8i8_to_v8i64(<8 x i64> addrspace(3)* %out, <8 x i8> addrspace(3)* %in) #0 {
  %load = load <8 x i8>, <8 x i8> addrspace(3)* %in
  %ext = sext <8 x i8> %load to <8 x i64>
  store <8 x i64> %ext, <8 x i64> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v16i8_to_v16i64:

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
define amdgpu_kernel void @local_zextload_v16i8_to_v16i64(<16 x i64> addrspace(3)* %out, <16 x i8> addrspace(3)* %in) #0 {
  %load = load <16 x i8>, <16 x i8> addrspace(3)* %in
  %ext = zext <16 x i8> %load to <16 x i64>
  store <16 x i64> %ext, <16 x i64> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v16i8_to_v16i64:

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
define amdgpu_kernel void @local_sextload_v16i8_to_v16i64(<16 x i64> addrspace(3)* %out, <16 x i8> addrspace(3)* %in) #0 {
  %load = load <16 x i8>, <16 x i8> addrspace(3)* %in
  %ext = sext <16 x i8> %load to <16 x i64>
  store <16 x i64> %ext, <16 x i64> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v32i8_to_v32i64:

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
define amdgpu_kernel void @local_zextload_v32i8_to_v32i64(<32 x i64> addrspace(3)* %out, <32 x i8> addrspace(3)* %in) #0 {
  %load = load <32 x i8>, <32 x i8> addrspace(3)* %in
  %ext = zext <32 x i8> %load to <32 x i64>
  store <32 x i64> %ext, <32 x i64> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v32i8_to_v32i64:

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
define amdgpu_kernel void @local_sextload_v32i8_to_v32i64(<32 x i64> addrspace(3)* %out, <32 x i8> addrspace(3)* %in) #0 {
  %load = load <32 x i8>, <32 x i8> addrspace(3)* %in
  %ext = sext <32 x i8> %load to <32 x i64>
  store <32 x i64> %ext, <32 x i64> addrspace(3)* %out
  ret void
}

; XFUNC-LABEL: {{^}}local_zextload_v64i8_to_v64i64:
; define amdgpu_kernel void @local_zextload_v64i8_to_v64i64(<64 x i64> addrspace(3)* %out, <64 x i8> addrspace(3)* %in) #0 {
;   %load = load <64 x i8>, <64 x i8> addrspace(3)* %in
;   %ext = zext <64 x i8> %load to <64 x i64>
;   store <64 x i64> %ext, <64 x i64> addrspace(3)* %out
;   ret void
; }

; XFUNC-LABEL: {{^}}local_sextload_v64i8_to_v64i64:
; define amdgpu_kernel void @local_sextload_v64i8_to_v64i64(<64 x i64> addrspace(3)* %out, <64 x i8> addrspace(3)* %in) #0 {
;   %load = load <64 x i8>, <64 x i8> addrspace(3)* %in
;   %ext = sext <64 x i8> %load to <64 x i64>
;   store <64 x i64> %ext, <64 x i64> addrspace(3)* %out
;   ret void
; }

; FUNC-LABEL: {{^}}local_zextload_i8_to_i16:
; GCN: ds_read_u8 v[[VAL:[0-9]+]],
; GCN: ds_write_b16 v[[VAL:[0-9]+]]

; EG: LDS_UBYTE_READ_RET
; EG: LDS_SHORT_WRITE
define amdgpu_kernel void @local_zextload_i8_to_i16(i16 addrspace(3)* %out, i8 addrspace(3)* %in) #0 {
  %a = load i8, i8 addrspace(3)* %in
  %ext = zext i8 %a to i16
  store i16 %ext, i16 addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_i8_to_i16:
; GCN: ds_read_i8 v[[VAL:[0-9]+]],
; GCN: ds_write_b16 v{{[0-9]+}}, v[[VAL]]

; EG: LDS_UBYTE_READ_RET
; EG: BFE_INT
; EG: LDS_SHORT_WRITE
define amdgpu_kernel void @local_sextload_i8_to_i16(i16 addrspace(3)* %out, i8 addrspace(3)* %in) #0 {
  %a = load i8, i8 addrspace(3)* %in
  %ext = sext i8 %a to i16
  store i16 %ext, i16 addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v1i8_to_v1i16:

; EG: LDS_UBYTE_READ_RET
; EG: LDS_SHORT_WRITE
define amdgpu_kernel void @local_zextload_v1i8_to_v1i16(<1 x i16> addrspace(3)* %out, <1 x i8> addrspace(3)* %in) #0 {
  %load = load <1 x i8>, <1 x i8> addrspace(3)* %in
  %ext = zext <1 x i8> %load to <1 x i16>
  store <1 x i16> %ext, <1 x i16> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v1i8_to_v1i16:

; EG: LDS_UBYTE_READ_RET
; EG: BFE_INT
; EG: LDS_SHORT_WRITE
define amdgpu_kernel void @local_sextload_v1i8_to_v1i16(<1 x i16> addrspace(3)* %out, <1 x i8> addrspace(3)* %in) #0 {
  %load = load <1 x i8>, <1 x i8> addrspace(3)* %in
  %ext = sext <1 x i8> %load to <1 x i16>
  store <1 x i16> %ext, <1 x i16> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v2i8_to_v2i16:

; EG: LDS_USHORT_READ_RET
; EG: LDS_WRITE
define amdgpu_kernel void @local_zextload_v2i8_to_v2i16(<2 x i16> addrspace(3)* %out, <2 x i8> addrspace(3)* %in) #0 {
  %load = load <2 x i8>, <2 x i8> addrspace(3)* %in
  %ext = zext <2 x i8> %load to <2 x i16>
  store <2 x i16> %ext, <2 x i16> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v2i8_to_v2i16:

; EG: LDS_USHORT_READ_RET
; EG: BFE_INT
; EG: BFE_INT
; EG: LDS_WRITE
define amdgpu_kernel void @local_sextload_v2i8_to_v2i16(<2 x i16> addrspace(3)* %out, <2 x i8> addrspace(3)* %in) #0 {
  %load = load <2 x i8>, <2 x i8> addrspace(3)* %in
  %ext = sext <2 x i8> %load to <2 x i16>
  store <2 x i16> %ext, <2 x i16> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v4i8_to_v4i16:

; EG: LDS_READ_RET
; EG: LDS_WRITE
; EG: LDS_WRITE
define amdgpu_kernel void @local_zextload_v4i8_to_v4i16(<4 x i16> addrspace(3)* %out, <4 x i8> addrspace(3)* %in) #0 {
  %load = load <4 x i8>, <4 x i8> addrspace(3)* %in
  %ext = zext <4 x i8> %load to <4 x i16>
  store <4 x i16> %ext, <4 x i16> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v4i8_to_v4i16:

; EG: LDS_READ_RET
; TODO: these do LSHR + BFE_INT, instead of just BFE_INT/ASHR
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG: LDS_WRITE
; EG: LDS_WRITE
define amdgpu_kernel void @local_sextload_v4i8_to_v4i16(<4 x i16> addrspace(3)* %out, <4 x i8> addrspace(3)* %in) #0 {
  %load = load <4 x i8>, <4 x i8> addrspace(3)* %in
  %ext = sext <4 x i8> %load to <4 x i16>
  store <4 x i16> %ext, <4 x i16> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v8i8_to_v8i16:

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
define amdgpu_kernel void @local_zextload_v8i8_to_v8i16(<8 x i16> addrspace(3)* %out, <8 x i8> addrspace(3)* %in) #0 {
  %load = load <8 x i8>, <8 x i8> addrspace(3)* %in
  %ext = zext <8 x i8> %load to <8 x i16>
  store <8 x i16> %ext, <8 x i16> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v8i8_to_v8i16:

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; TODO: these do LSHR + BFE_INT, instead of just BFE_INT/ASHR
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
define amdgpu_kernel void @local_sextload_v8i8_to_v8i16(<8 x i16> addrspace(3)* %out, <8 x i8> addrspace(3)* %in) #0 {
  %load = load <8 x i8>, <8 x i8> addrspace(3)* %in
  %ext = sext <8 x i8> %load to <8 x i16>
  store <8 x i16> %ext, <8 x i16> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v16i8_to_v16i16:

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
define amdgpu_kernel void @local_zextload_v16i8_to_v16i16(<16 x i16> addrspace(3)* %out, <16 x i8> addrspace(3)* %in) #0 {
  %load = load <16 x i8>, <16 x i8> addrspace(3)* %in
  %ext = zext <16 x i8> %load to <16 x i16>
  store <16 x i16> %ext, <16 x i16> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v16i8_to_v16i16:

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; TODO: these do LSHR + BFE_INT, instead of just BFE_INT/ASHR
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
define amdgpu_kernel void @local_sextload_v16i8_to_v16i16(<16 x i16> addrspace(3)* %out, <16 x i8> addrspace(3)* %in) #0 {
  %load = load <16 x i8>, <16 x i8> addrspace(3)* %in
  %ext = sext <16 x i8> %load to <16 x i16>
  store <16 x i16> %ext, <16 x i16> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v32i8_to_v32i16:

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
define amdgpu_kernel void @local_zextload_v32i8_to_v32i16(<32 x i16> addrspace(3)* %out, <32 x i8> addrspace(3)* %in) #0 {
  %load = load <32 x i8>, <32 x i8> addrspace(3)* %in
  %ext = zext <32 x i8> %load to <32 x i16>
  store <32 x i16> %ext, <32 x i16> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v32i8_to_v32i16:

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; TODO: these do LSHR + BFE_INT, instead of just BFE_INT/ASHR
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
; EG: LDS_WRITE
define amdgpu_kernel void @local_sextload_v32i8_to_v32i16(<32 x i16> addrspace(3)* %out, <32 x i8> addrspace(3)* %in) #0 {
  %load = load <32 x i8>, <32 x i8> addrspace(3)* %in
  %ext = sext <32 x i8> %load to <32 x i16>
  store <32 x i16> %ext, <32 x i16> addrspace(3)* %out
  ret void
}

; XFUNC-LABEL: {{^}}local_zextload_v64i8_to_v64i16:
; define amdgpu_kernel void @local_zextload_v64i8_to_v64i16(<64 x i16> addrspace(3)* %out, <64 x i8> addrspace(3)* %in) #0 {
;   %load = load <64 x i8>, <64 x i8> addrspace(3)* %in
;   %ext = zext <64 x i8> %load to <64 x i16>
;   store <64 x i16> %ext, <64 x i16> addrspace(3)* %out
;   ret void
; }

; XFUNC-LABEL: {{^}}local_sextload_v64i8_to_v64i16:
; define amdgpu_kernel void @local_sextload_v64i8_to_v64i16(<64 x i16> addrspace(3)* %out, <64 x i8> addrspace(3)* %in) #0 {
;   %load = load <64 x i8>, <64 x i8> addrspace(3)* %in
;   %ext = sext <64 x i8> %load to <64 x i16>
;   store <64 x i16> %ext, <64 x i16> addrspace(3)* %out
;   ret void
; }

attributes #0 = { nounwind }
