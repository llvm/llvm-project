; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -allow-deprecated-dag-overlap -check-prefixes=GCN,SI,SICIVI,FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -allow-deprecated-dag-overlap -check-prefixes=GCN,SICIVI,GFX89,FUNC %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -allow-deprecated-dag-overlap -check-prefixes=GCN,GFX9,GFX89,FUNC %s
; RUN: llc -march=r600 -mcpu=redwood -verify-machineinstrs < %s | FileCheck -allow-deprecated-dag-overlap -check-prefix=EG -check-prefix=FUNC %s

; Testing for ds_read/write_b128
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=+enable-ds128 < %s | FileCheck -allow-deprecated-dag-overlap -check-prefixes=CIVI,FUNC %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -mattr=+enable-ds128 < %s | FileCheck -allow-deprecated-dag-overlap -check-prefixes=CIVI,FUNC %s

; FUNC-LABEL: {{^}}local_load_i16:
; GFX9-NOT: m0
; SICIVI: s_mov_b32 m0

; GCN: ds_read_u16 v{{[0-9]+}}

; EG: MOV {{[* ]*}}[[FROM:T[0-9]+\.[XYZW]]], KC0[2].Z
; EG: LDS_USHORT_READ_RET {{.*}} [[FROM]]
; EG-DAG: MOV {{[* ]*}}[[DATA:T[0-9]+\.[XYZW]]], OQAP
; EG-DAG: MOV {{[* ]*}}[[TO:T[0-9]+\.[XYZW]]], KC0[2].Y
; EG: LDS_SHORT_WRITE {{\*?}} [[TO]], [[DATA]]
define amdgpu_kernel void @local_load_i16(i16 addrspace(3)* %out, i16 addrspace(3)* %in) {
entry:
  %ld = load i16, i16 addrspace(3)* %in
  store i16 %ld, i16 addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_load_v2i16:
; GFX9-NOT: m0
; SICIVI: s_mov_b32 m0

; GCN: ds_read_b32

; EG: MOV {{[* ]*}}[[FROM:T[0-9]+\.[XYZW]]], KC0[2].Z
; EG: LDS_READ_RET {{.*}} [[FROM]]
; EG-DAG: MOV {{[* ]*}}[[DATA:T[0-9]+\.[XYZW]]], OQAP
; EG-DAG: MOV {{[* ]*}}[[TO:T[0-9]+\.[XYZW]]], KC0[2].Y
; EG: LDS_WRITE {{\*?}} [[TO]], [[DATA]]
define amdgpu_kernel void @local_load_v2i16(<2 x i16> addrspace(3)* %out, <2 x i16> addrspace(3)* %in) {
entry:
  %ld = load <2 x i16>, <2 x i16> addrspace(3)* %in
  store <2 x i16> %ld, <2 x i16> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_load_v3i16:
; GFX9-NOT: m0
; SICIVI: s_mov_b32 m0

; GCN: ds_read_b64
; GCN-DAG: ds_write_b32
; GCN-DAG: ds_write_b16

; EG-DAG: LDS_USHORT_READ_RET
; EG-DAG: LDS_READ_RET
define amdgpu_kernel void @local_load_v3i16(<3 x i16> addrspace(3)* %out, <3 x i16> addrspace(3)* %in) {
entry:
  %ld = load <3 x i16>, <3 x i16> addrspace(3)* %in
  store <3 x i16> %ld, <3 x i16> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_load_v4i16:
; GFX9-NOT: m0
; SICIVI: s_mov_b32 m0

; GCN: ds_read_b64

; EG: LDS_READ_RET
; EG: LDS_READ_RET
define amdgpu_kernel void @local_load_v4i16(<4 x i16> addrspace(3)* %out, <4 x i16> addrspace(3)* %in) {
entry:
  %ld = load <4 x i16>, <4 x i16> addrspace(3)* %in
  store <4 x i16> %ld, <4 x i16> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_load_v8i16:
; GFX9-NOT: m0
; SICIVI: s_mov_b32 m0

; GCN: ds_read2_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset1:1{{$}}

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
define amdgpu_kernel void @local_load_v8i16(<8 x i16> addrspace(3)* %out, <8 x i16> addrspace(3)* %in) {
entry:
  %ld = load <8 x i16>, <8 x i16> addrspace(3)* %in
  store <8 x i16> %ld, <8 x i16> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_load_v16i16:
; GFX9-NOT: m0
; SICIVI: s_mov_b32 m0

; GCN-DAG: ds_read2_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset0:2 offset1:3{{$}}
; GCN-DAG: ds_read2_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset1:1{{$}}


; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
define amdgpu_kernel void @local_load_v16i16(<16 x i16> addrspace(3)* %out, <16 x i16> addrspace(3)* %in) {
entry:
  %ld = load <16 x i16>, <16 x i16> addrspace(3)* %in
  store <16 x i16> %ld, <16 x i16> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_i16_to_i32:
; GFX9-NOT: m0
; SICIVI: s_mov_b32 m0

; GCN: ds_read_u16
; GCN: ds_write_b32

; EG: MOV {{[* ]*}}[[FROM:T[0-9]+\.[XYZW]]], KC0[2].Z
; EG: LDS_USHORT_READ_RET {{.*}} [[FROM]]
; EG-DAG: MOV {{[* ]*}}[[DATA:T[0-9]+\.[XYZW]]], OQAP
; EG-DAG: MOV {{[* ]*}}[[TO:T[0-9]+\.[XYZW]]], KC0[2].Y
; EG: LDS_WRITE {{\*?}} [[TO]], [[DATA]]
define amdgpu_kernel void @local_zextload_i16_to_i32(i32 addrspace(3)* %out, i16 addrspace(3)* %in) #0 {
  %a = load i16, i16 addrspace(3)* %in
  %ext = zext i16 %a to i32
  store i32 %ext, i32 addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_i16_to_i32:
; GCN-NOT: s_wqm_b64

; GFX9-NOT: m0
; SICIVI: s_mov_b32 m0

; GCN: ds_read_i16

; EG: MOV {{[* ]*}}[[FROM:T[0-9]+\.[XYZW]]], KC0[2].Z
; EG: LDS_USHORT_READ_RET {{.*}} [[FROM]]
; EG-DAG: MOV {{[* ]*}}[[TMP:T[0-9]+\.[XYZW]]], OQAP
; EG-DAG: MOV {{[* ]*}}[[TO:T[0-9]+\.[XYZW]]], KC0[2].Y
; EG-DAG: BFE_INT {{[* ]*}}[[DATA:T[0-9]+\.[XYZW]]], {{.*}}, 0.0, literal
; EG: 16
; EG: LDS_WRITE {{\*?}} [[TO]], [[DATA]]
define amdgpu_kernel void @local_sextload_i16_to_i32(i32 addrspace(3)* %out, i16 addrspace(3)* %in) #0 {
  %a = load i16, i16 addrspace(3)* %in
  %ext = sext i16 %a to i32
  store i32 %ext, i32 addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v1i16_to_v1i32:
; GFX9-NOT: m0
; SICIVI: s_mov_b32 m0

; GCN: ds_read_u16

; EG: MOV {{[* ]*}}[[FROM:T[0-9]+\.[XYZW]]], KC0[2].Z
; EG: LDS_USHORT_READ_RET {{.*}} [[FROM]]
; EG-DAG: MOV {{[* ]*}}[[DATA:T[0-9]+\.[XYZW]]], OQAP
; EG-DAG: MOV {{[* ]*}}[[TO:T[0-9]+\.[XYZW]]], KC0[2].Y
; EG: LDS_WRITE {{\*?}} [[TO]], [[DATA]]
define amdgpu_kernel void @local_zextload_v1i16_to_v1i32(<1 x i32> addrspace(3)* %out, <1 x i16> addrspace(3)* %in) #0 {
  %load = load <1 x i16>, <1 x i16> addrspace(3)* %in
  %ext = zext <1 x i16> %load to <1 x i32>
  store <1 x i32> %ext, <1 x i32> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v1i16_to_v1i32:
; GFX9-NOT: m0
; SICIVI: s_mov_b32 m0

; GCN: ds_read_i16

; EG: MOV {{[* ]*}}[[FROM:T[0-9]+\.[XYZW]]], KC0[2].Z
; EG: LDS_USHORT_READ_RET {{.*}} [[FROM]]
; EG-DAG: MOV {{[* ]*}}[[TMP:T[0-9]+\.[XYZW]]], OQAP
; EG-DAG: MOV {{[* ]*}}[[TO:T[0-9]+\.[XYZW]]], KC0[2].Y
; EG-DAG: BFE_INT {{[* ]*}}[[DATA:T[0-9]+\.[XYZW]]], {{.*}}, 0.0, literal
; EG: 16
; EG: LDS_WRITE {{\*?}} [[TO]], [[DATA]]
define amdgpu_kernel void @local_sextload_v1i16_to_v1i32(<1 x i32> addrspace(3)* %out, <1 x i16> addrspace(3)* %in) #0 {
  %load = load <1 x i16>, <1 x i16> addrspace(3)* %in
  %ext = sext <1 x i16> %load to <1 x i32>
  store <1 x i32> %ext, <1 x i32> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v2i16_to_v2i32:
; GCN-NOT: s_wqm_b64
; GFX9-NOT: m0
; SICIVI: s_mov_b32 m0

; GCN: ds_read_b32

; EG: LDS_READ_RET
define amdgpu_kernel void @local_zextload_v2i16_to_v2i32(<2 x i32> addrspace(3)* %out, <2 x i16> addrspace(3)* %in) #0 {
  %load = load <2 x i16>, <2 x i16> addrspace(3)* %in
  %ext = zext <2 x i16> %load to <2 x i32>
  store <2 x i32> %ext, <2 x i32> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v2i16_to_v2i32:
; GCN-NOT: s_wqm_b64
; GFX9-NOT: m0
; SICIVI: s_mov_b32 m0

; GCN: ds_read_b32

; EG: LDS_READ_RET
; EG: BFE_INT
; EG: BFE_INT
define amdgpu_kernel void @local_sextload_v2i16_to_v2i32(<2 x i32> addrspace(3)* %out, <2 x i16> addrspace(3)* %in) #0 {
  %load = load <2 x i16>, <2 x i16> addrspace(3)* %in
  %ext = sext <2 x i16> %load to <2 x i32>
  store <2 x i32> %ext, <2 x i32> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_local_zextload_v3i16_to_v3i32:
; GFX9-NOT: m0
; SICIVI: s_mov_b32 m0

; GCN: ds_read_b64
; GCN-DAG: ds_write_b32
; GCN-DAG: ds_write_b64

; EG: LDS_READ_RET
define amdgpu_kernel void @local_local_zextload_v3i16_to_v3i32(<3 x i32> addrspace(3)* %out, <3 x i16> addrspace(3)* %in) {
entry:
  %ld = load <3 x i16>, <3 x i16> addrspace(3)* %in
  %ext = zext <3 x i16> %ld to <3 x i32>
  store <3 x i32> %ext, <3 x i32> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_local_sextload_v3i16_to_v3i32:
; GFX9-NOT: m0
; SICIVI: s_mov_b32 m0

; GCN: ds_read_b64
; GCN-DAG: ds_write_b32
; GCN-DAG: ds_write_b64

; EG: LDS_READ_RET
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
define amdgpu_kernel void @local_local_sextload_v3i16_to_v3i32(<3 x i32> addrspace(3)* %out, <3 x i16> addrspace(3)* %in) {
entry:
  %ld = load <3 x i16>, <3 x i16> addrspace(3)* %in
  %ext = sext <3 x i16> %ld to <3 x i32>
  store <3 x i32> %ext, <3 x i32> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_local_zextload_v4i16_to_v4i32:
; GCN-NOT: s_wqm_b64
; GFX9-NOT: m0
; SICIVI: s_mov_b32 m0

; GCN: ds_read_b64

; EG: LDS_READ_RET
; EG: LDS_READ_RET
define amdgpu_kernel void @local_local_zextload_v4i16_to_v4i32(<4 x i32> addrspace(3)* %out, <4 x i16> addrspace(3)* %in) #0 {
  %load = load <4 x i16>, <4 x i16> addrspace(3)* %in
  %ext = zext <4 x i16> %load to <4 x i32>
  store <4 x i32> %ext, <4 x i32> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v4i16_to_v4i32:
; GCN-NOT: s_wqm_b64
; GFX9-NOT: m0
; SICIVI: s_mov_b32 m0

; GCN: ds_read_b64

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
define amdgpu_kernel void @local_sextload_v4i16_to_v4i32(<4 x i32> addrspace(3)* %out, <4 x i16> addrspace(3)* %in) #0 {
  %load = load <4 x i16>, <4 x i16> addrspace(3)* %in
  %ext = sext <4 x i16> %load to <4 x i32>
  store <4 x i32> %ext, <4 x i32> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v8i16_to_v8i32:
; GFX9-NOT: m0
; SICIVI: s_mov_b32 m0

; GCN: ds_read2_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset1:1{{$}}

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
define amdgpu_kernel void @local_zextload_v8i16_to_v8i32(<8 x i32> addrspace(3)* %out, <8 x i16> addrspace(3)* %in) #0 {
  %load = load <8 x i16>, <8 x i16> addrspace(3)* %in
  %ext = zext <8 x i16> %load to <8 x i32>
  store <8 x i32> %ext, <8 x i32> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v8i16_to_v8i32:
; GFX9-NOT: m0
; SICIVI: s_mov_b32 m0

; GCN: ds_read2_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset1:1{{$}}

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
define amdgpu_kernel void @local_sextload_v8i16_to_v8i32(<8 x i32> addrspace(3)* %out, <8 x i16> addrspace(3)* %in) #0 {
  %load = load <8 x i16>, <8 x i16> addrspace(3)* %in
  %ext = sext <8 x i16> %load to <8 x i32>
  store <8 x i32> %ext, <8 x i32> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v16i16_to_v16i32:
; GFX9-NOT: m0
; SICIVI: s_mov_b32 m0

; GCN-DAG: ds_read2_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset1:1{{$}}
; GCN-DAG: ds_read2_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset0:2 offset1:3{{$}}

; GCN: ds_write2_b64
; GCN: ds_write2_b64
; GCN: ds_write2_b64
; GCN: ds_write2_b64

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
define amdgpu_kernel void @local_zextload_v16i16_to_v16i32(<16 x i32> addrspace(3)* %out, <16 x i16> addrspace(3)* %in) #0 {
  %load = load <16 x i16>, <16 x i16> addrspace(3)* %in
  %ext = zext <16 x i16> %load to <16 x i32>
  store <16 x i32> %ext, <16 x i32> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v16i16_to_v16i32:
; GFX9-NOT: m0
; SICIVI: s_mov_b32 m0


; GCN-DAG: ds_read2_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset1:1{{$}}
; GCN-DAG: ds_read2_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset0:2 offset1:3{{$}}

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
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
define amdgpu_kernel void @local_sextload_v16i16_to_v16i32(<16 x i32> addrspace(3)* %out, <16 x i16> addrspace(3)* %in) #0 {
  %load = load <16 x i16>, <16 x i16> addrspace(3)* %in
  %ext = sext <16 x i16> %load to <16 x i32>
  store <16 x i32> %ext, <16 x i32> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v32i16_to_v32i32:
; GFX9-NOT: m0
; SICIVI: s_mov_b32 m0

; GCN-DAG: ds_read2_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset1:1{{$}}
; GCN-DAG: ds_read2_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset0:2 offset1:3
; GCN-DAG: ds_read2_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset0:4 offset1:5
; GCN-DAG: ds_read2_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset0:6 offset1:7

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
define amdgpu_kernel void @local_zextload_v32i16_to_v32i32(<32 x i32> addrspace(3)* %out, <32 x i16> addrspace(3)* %in) #0 {
  %load = load <32 x i16>, <32 x i16> addrspace(3)* %in
  %ext = zext <32 x i16> %load to <32 x i32>
  store <32 x i32> %ext, <32 x i32> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v32i16_to_v32i32:
; GFX9-NOT: m0
; SICIVI: s_mov_b32 m0

; GCN-DAG: ds_read2_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset0:2 offset1:3{{$}}
; GCN-DAG: ds_read2_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset0:4 offset1:5
; GCN-DAG: ds_read2_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset1:1{{$}}
; GCN-DAG: ds_read2_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset0:6 offset1:7
; GCN-DAG: ds_write2_b64 v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}} offset0:14 offset1:15
; GCN-DAG: ds_write2_b64 v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}} offset0:12 offset1:13
; GCN-DAG: ds_write2_b64 v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}} offset0:10 offset1:11
; GCN-DAG: ds_write2_b64 v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}} offset0:8 offset1:9
; GCN-DAG: ds_write2_b64 v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}} offset0:6 offset1:7
; GCN-DAG: ds_write2_b64 v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}} offset0:4 offset1:5
; GCN-DAG: ds_write2_b64 v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}} offset0:2 offset1:3
; GCN-DAG: ds_write2_b64 v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}} offset1:1

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
define amdgpu_kernel void @local_sextload_v32i16_to_v32i32(<32 x i32> addrspace(3)* %out, <32 x i16> addrspace(3)* %in) #0 {
  %load = load <32 x i16>, <32 x i16> addrspace(3)* %in
  %ext = sext <32 x i16> %load to <32 x i32>
  store <32 x i32> %ext, <32 x i32> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v64i16_to_v64i32:
; GFX9-NOT: m0
; SICIVI: s_mov_b32 m0

; GCN-DAG: ds_read2_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset0:14 offset1:15
; GCN-DAG: ds_read2_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset1:1{{$}}
; GCN-DAG: ds_read2_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset0:2 offset1:3
; GCN-DAG: ds_read2_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset0:4 offset1:5
; GCN-DAG: ds_read2_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset0:6 offset1:7
; GCN-DAG: ds_read2_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset0:8 offset1:9
; GCN-DAG: ds_read2_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset0:12 offset1:13
; GCN-DAG: ds_read2_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset0:10 offset1:11
; GCN-DAG: ds_write2_b64 v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}} offset0:30 offset1:31
; GCN-DAG: ds_write2_b64 v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}} offset0:28 offset1:29
; GCN-DAG: ds_write2_b64 v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}} offset0:26 offset1:27
; GCN-DAG: ds_write2_b64 v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}} offset0:24 offset1:25
; GCN-DAG: ds_write2_b64 v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}} offset0:22 offset1:23
; GCN-DAG: ds_write2_b64 v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}} offset0:20 offset1:21
; GCN-DAG: ds_write2_b64 v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}} offset0:18 offset1:19
; GCN-DAG: ds_write2_b64 v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}} offset0:16 offset1:17
; GCN-DAG: ds_write2_b64 v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}} offset0:14 offset1:15
; GCN-DAG: ds_write2_b64 v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}} offset0:12 offset1:13
; GCN-DAG: ds_write2_b64 v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}} offset0:10 offset1:11
; GCN-DAG: ds_write2_b64 v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}} offset0:8 offset1:9
; GCN-DAG: ds_write2_b64 v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}} offset0:6 offset1:7
; GCN-DAG: ds_write2_b64 v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}} offset0:4 offset1:5
; GCN-DAG: ds_write2_b64 v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}} offset0:2 offset1:3
; GCN-DAG: ds_write2_b64 v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}} offset1:1

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
define amdgpu_kernel void @local_zextload_v64i16_to_v64i32(<64 x i32> addrspace(3)* %out, <64 x i16> addrspace(3)* %in) #0 {
  %load = load <64 x i16>, <64 x i16> addrspace(3)* %in
  %ext = zext <64 x i16> %load to <64 x i32>
  store <64 x i32> %ext, <64 x i32> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v64i16_to_v64i32:
; GFX9-NOT: m0
; SICIVI: s_mov_b32 m0

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
define amdgpu_kernel void @local_sextload_v64i16_to_v64i32(<64 x i32> addrspace(3)* %out, <64 x i16> addrspace(3)* %in) #0 {
  %load = load <64 x i16>, <64 x i16> addrspace(3)* %in
  %ext = sext <64 x i16> %load to <64 x i32>
  store <64 x i32> %ext, <64 x i32> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_i16_to_i64:
; GFX9-NOT: m0
; SICIVI: s_mov_b32 m0

; GCN-DAG: ds_read_u16 v[[LO:[0-9]+]],
; GCN-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], 0{{$}}

; GCN: ds_write_b64 v{{[0-9]+}}, v{{\[}}[[LO]]:[[HI]]]

; EG: MOV {{[* ]*}}[[FROM:T[0-9]+\.[XYZW]]], KC0[2].Z
; EG: LDS_USHORT_READ_RET {{.*}} [[FROM]]
; EG-DAG: MOV {{[* ]*}}[[DATA:T[0-9]+\.[XYZW]]], OQAP
; EG-DAG: MOV {{[* ]*}}[[TO:T[0-9]+\.[XYZW]]], KC0[2].Y
; EG-DAG: LDS_WRITE
define amdgpu_kernel void @local_zextload_i16_to_i64(i64 addrspace(3)* %out, i16 addrspace(3)* %in) #0 {
  %a = load i16, i16 addrspace(3)* %in
  %ext = zext i16 %a to i64
  store i64 %ext, i64 addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_i16_to_i64:
; GFX9-NOT: m0
; SICIVI: s_mov_b32 m0

; FIXME: Need to optimize this sequence to avoid an extra shift.
;  t25: i32,ch = load<LD2[%in(addrspace=3)], anyext from i16> t12, t10, undef:i32
;          t28: i64 = any_extend t25
;        t30: i64 = sign_extend_inreg t28, ValueType:ch:i16
; SI: ds_read_i16 v[[LO:[0-9]+]],
; GFX89: ds_read_u16 v[[ULO:[0-9]+]]
; GFX89: v_bfe_i32 v[[LO:[0-9]+]], v[[ULO]], 0, 16
; GCN-DAG: v_ashrrev_i32_e32 v[[HI:[0-9]+]], 31, v[[LO]]

; GCN: ds_write_b64 v{{[0-9]+}}, v{{\[}}[[LO]]:[[HI]]]

; EG: MOV {{[* ]*}}[[FROM:T[0-9]+\.[XYZW]]], KC0[2].Z
; EG: LDS_USHORT_READ_RET {{.*}} [[FROM]]
; EG-DAG: MOV {{[* ]*}}[[TMP:T[0-9]+\.[XYZW]]], OQAP
; EG-DAG: MOV {{[* ]*}}[[TO:T[0-9]+\.[XYZW]]], KC0[2].Y
; EG-DAG: BFE_INT {{[* ]*}}[[DATA:T[0-9]+\.[XYZW]]], {{.*}}, 0.0, literal
; EG-DAG: LDS_WRITE
; EG-DAG: 16
; EG: LDS_WRITE {{\*?}} [[TO]], [[DATA]]
define amdgpu_kernel void @local_sextload_i16_to_i64(i64 addrspace(3)* %out, i16 addrspace(3)* %in) #0 {
  %a = load i16, i16 addrspace(3)* %in
  %ext = sext i16 %a to i64
  store i64 %ext, i64 addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v1i16_to_v1i64:
; GFX9-NOT: m0
; SICIVI: s_mov_b32 m0


; EG: MOV {{[* ]*}}[[FROM:T[0-9]+\.[XYZW]]], KC0[2].Z
; EG: LDS_USHORT_READ_RET {{.*}} [[FROM]]
; EG-DAG: MOV {{[* ]*}}[[DATA:T[0-9]+\.[XYZW]]], OQAP
; EG-DAG: MOV {{[* ]*}}[[TO:T[0-9]+\.[XYZW]]], KC0[2].Y
; EG-DAG: LDS_WRITE
define amdgpu_kernel void @local_zextload_v1i16_to_v1i64(<1 x i64> addrspace(3)* %out, <1 x i16> addrspace(3)* %in) #0 {
  %load = load <1 x i16>, <1 x i16> addrspace(3)* %in
  %ext = zext <1 x i16> %load to <1 x i64>
  store <1 x i64> %ext, <1 x i64> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v1i16_to_v1i64:
; GFX9-NOT: m0
; SICIVI: s_mov_b32 m0


; EG: MOV {{[* ]*}}[[FROM:T[0-9]+\.[XYZW]]], KC0[2].Z
; EG: LDS_USHORT_READ_RET {{.*}} [[FROM]]
; EG-DAG: MOV {{[* ]*}}[[TMP:T[0-9]+\.[XYZW]]], OQAP
; EG-DAG: MOV {{[* ]*}}[[TO:T[0-9]+\.[XYZW]]], KC0[2].Y
; EG-DAG: BFE_INT {{[* ]*}}[[DATA:T[0-9]+\.[XYZW]]], {{.*}}, 0.0, literal
; EG-DAG: LDS_WRITE
; EG-DAG: 16
; EG: LDS_WRITE {{\*?}} [[TO]], [[DATA]]
define amdgpu_kernel void @local_sextload_v1i16_to_v1i64(<1 x i64> addrspace(3)* %out, <1 x i16> addrspace(3)* %in) #0 {
  %load = load <1 x i16>, <1 x i16> addrspace(3)* %in
  %ext = sext <1 x i16> %load to <1 x i64>
  store <1 x i64> %ext, <1 x i64> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v2i16_to_v2i64:
; GFX9-NOT: m0
; SICIVI: s_mov_b32 m0


; EG: LDS_READ_RET
define amdgpu_kernel void @local_zextload_v2i16_to_v2i64(<2 x i64> addrspace(3)* %out, <2 x i16> addrspace(3)* %in) #0 {
  %load = load <2 x i16>, <2 x i16> addrspace(3)* %in
  %ext = zext <2 x i16> %load to <2 x i64>
  store <2 x i64> %ext, <2 x i64> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v2i16_to_v2i64:
; GFX9-NOT: m0
; SICIVI: s_mov_b32 m0


; EG: LDS_READ_RET
; EG-DAG: BFE_INT
; EG-DAG: ASHR
define amdgpu_kernel void @local_sextload_v2i16_to_v2i64(<2 x i64> addrspace(3)* %out, <2 x i16> addrspace(3)* %in) #0 {
  %load = load <2 x i16>, <2 x i16> addrspace(3)* %in
  %ext = sext <2 x i16> %load to <2 x i64>
  store <2 x i64> %ext, <2 x i64> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v4i16_to_v4i64:
; GFX9-NOT: m0
; SICIVI: s_mov_b32 m0


; EG: LDS_READ_RET
; EG: LDS_READ_RET
define amdgpu_kernel void @local_zextload_v4i16_to_v4i64(<4 x i64> addrspace(3)* %out, <4 x i16> addrspace(3)* %in) #0 {
  %load = load <4 x i16>, <4 x i16> addrspace(3)* %in
  %ext = zext <4 x i16> %load to <4 x i64>
  store <4 x i64> %ext, <4 x i64> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v4i16_to_v4i64:
; GFX9-NOT: m0
; SICIVI: s_mov_b32 m0


; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: ASHR
; EG-DAG: ASHR
define amdgpu_kernel void @local_sextload_v4i16_to_v4i64(<4 x i64> addrspace(3)* %out, <4 x i16> addrspace(3)* %in) #0 {
  %load = load <4 x i16>, <4 x i16> addrspace(3)* %in
  %ext = sext <4 x i16> %load to <4 x i64>
  store <4 x i64> %ext, <4 x i64> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v8i16_to_v8i64:
; GFX9-NOT: m0
; SICIVI: s_mov_b32 m0


; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
define amdgpu_kernel void @local_zextload_v8i16_to_v8i64(<8 x i64> addrspace(3)* %out, <8 x i16> addrspace(3)* %in) #0 {
  %load = load <8 x i16>, <8 x i16> addrspace(3)* %in
  %ext = zext <8 x i16> %load to <8 x i64>
  store <8 x i64> %ext, <8 x i64> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v8i16_to_v8i64:
; GFX9-NOT: m0
; SICIVI: s_mov_b32 m0


; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: ASHR
; EG-DAG: ASHR
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: ASHR
; EG-DAG: ASHR
define amdgpu_kernel void @local_sextload_v8i16_to_v8i64(<8 x i64> addrspace(3)* %out, <8 x i16> addrspace(3)* %in) #0 {
  %load = load <8 x i16>, <8 x i16> addrspace(3)* %in
  %ext = sext <8 x i16> %load to <8 x i64>
  store <8 x i64> %ext, <8 x i64> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v16i16_to_v16i64:
; GFX9-NOT: m0
; SICIVI: s_mov_b32 m0


; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
define amdgpu_kernel void @local_zextload_v16i16_to_v16i64(<16 x i64> addrspace(3)* %out, <16 x i16> addrspace(3)* %in) #0 {
  %load = load <16 x i16>, <16 x i16> addrspace(3)* %in
  %ext = zext <16 x i16> %load to <16 x i64>
  store <16 x i64> %ext, <16 x i64> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v16i16_to_v16i64:
; GFX9-NOT: m0
; SICIVI: s_mov_b32 m0


; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: ASHR
; EG-DAG: ASHR
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: ASHR
; EG-DAG: ASHR
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: ASHR
; EG-DAG: ASHR
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: ASHR
; EG-DAG: ASHR
define amdgpu_kernel void @local_sextload_v16i16_to_v16i64(<16 x i64> addrspace(3)* %out, <16 x i16> addrspace(3)* %in) #0 {
  %load = load <16 x i16>, <16 x i16> addrspace(3)* %in
  %ext = sext <16 x i16> %load to <16 x i64>
  store <16 x i64> %ext, <16 x i64> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v32i16_to_v32i64:
; GFX9-NOT: m0
; SICIVI: s_mov_b32 m0


; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
define amdgpu_kernel void @local_zextload_v32i16_to_v32i64(<32 x i64> addrspace(3)* %out, <32 x i16> addrspace(3)* %in) #0 {
  %load = load <32 x i16>, <32 x i16> addrspace(3)* %in
  %ext = zext <32 x i16> %load to <32 x i64>
  store <32 x i64> %ext, <32 x i64> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v32i16_to_v32i64:
; GFX9-NOT: m0
; SICIVI: s_mov_b32 m0


; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: ASHR
; EG-DAG: ASHR
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: ASHR
; EG-DAG: ASHR
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: ASHR
; EG-DAG: ASHR
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: ASHR
; EG-DAG: ASHR
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: ASHR
; EG-DAG: ASHR
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: ASHR
; EG-DAG: ASHR
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: ASHR
; EG-DAG: ASHR
; EG-DAG: BFE_INT
; EG-DAG: BFE_INT
; EG-DAG: ASHR
; EG-DAG: ASHR
define amdgpu_kernel void @local_sextload_v32i16_to_v32i64(<32 x i64> addrspace(3)* %out, <32 x i16> addrspace(3)* %in) #0 {
  %load = load <32 x i16>, <32 x i16> addrspace(3)* %in
  %ext = sext <32 x i16> %load to <32 x i64>
  store <32 x i64> %ext, <32 x i64> addrspace(3)* %out
  ret void
}

; ; XFUNC-LABEL: {{^}}local_zextload_v64i16_to_v64i64:
; define amdgpu_kernel void @local_zextload_v64i16_to_v64i64(<64 x i64> addrspace(3)* %out, <64 x i16> addrspace(3)* %in) #0 {
;   %load = load <64 x i16>, <64 x i16> addrspace(3)* %in
;   %ext = zext <64 x i16> %load to <64 x i64>
;   store <64 x i64> %ext, <64 x i64> addrspace(3)* %out
;   ret void
; }

; ; XFUNC-LABEL: {{^}}local_sextload_v64i16_to_v64i64:
; define amdgpu_kernel void @local_sextload_v64i16_to_v64i64(<64 x i64> addrspace(3)* %out, <64 x i16> addrspace(3)* %in) #0 {
;   %load = load <64 x i16>, <64 x i16> addrspace(3)* %in
;   %ext = sext <64 x i16> %load to <64 x i64>
;   store <64 x i64> %ext, <64 x i64> addrspace(3)* %out
;   ret void
; }

; Tests if ds_read/write_b128 gets generated for the 16 byte aligned load.
; FUNC-LABEL: {{^}}local_v8i16_to_128:

; SI-NOT: ds_read_b128
; SI-NOT: ds_write_b128

; CIVI: ds_read_b128
; CIVI: ds_write_b128

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
define amdgpu_kernel void @local_v8i16_to_128(<8 x i16> addrspace(3)* %out, <8 x i16> addrspace(3)* %in) {
  %ld = load <8 x i16>, <8 x i16> addrspace(3)* %in, align 16
  store <8 x i16> %ld, <8 x i16> addrspace(3)* %out, align 16
  ret void
}

attributes #0 = { nounwind }
