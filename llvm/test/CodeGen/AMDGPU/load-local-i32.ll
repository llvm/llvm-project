; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SICIVI,FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global,-enable-ds128 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SICIVI,FUNC %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -mattr=-flat-for-global,-enable-ds128 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,VI,FUNC %s
; RUN: llc -march=amdgcn -mcpu=gfx908 -mattr=-flat-for-global,-enable-ds128 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,VI,FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; Testing for ds_read/write_128
; RUN: llc -march=amdgcn -mcpu=tahiti -mattr=+enable-ds128 < %s | FileCheck -check-prefixes=SI,FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=+enable-ds128 < %s | FileCheck -check-prefixes=CIVI,FUNC %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -mattr=+enable-ds128 < %s | FileCheck -check-prefixes=CIVI,FUNC %s

; FUNC-LABEL: {{^}}local_load_i32:
; GCN-NOT: s_wqm_b64
; SICIVI: s_mov_b32 m0, -1
; GFX9-NOT: m0
; GCN: ds_read_b32

; EG: LDS_READ_RET
define amdgpu_kernel void @local_load_i32(i32 addrspace(3)* %out, i32 addrspace(3)* %in) #0 {
entry:
  %ld = load i32, i32 addrspace(3)* %in
  store i32 %ld, i32 addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_load_v2i32:
; SICIVI: s_mov_b32 m0, -1
; GFX9-NOT: m0

; GCN: ds_read_b64
define amdgpu_kernel void @local_load_v2i32(<2 x i32> addrspace(3)* %out, <2 x i32> addrspace(3)* %in) #0 {
entry:
  %ld = load <2 x i32>, <2 x i32> addrspace(3)* %in
  store <2 x i32> %ld, <2 x i32> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_load_v3i32:
; SICIVI: s_mov_b32 m0, -1
; GFX9-NOT: m0

; SI-DAG: ds_read_b64
; SI-DAG: ds_read_b32
; CIVI-DAG: ds_read_b96
define amdgpu_kernel void @local_load_v3i32(<3 x i32> addrspace(3)* %out, <3 x i32> addrspace(3)* %in) #0 {
entry:
  %ld = load <3 x i32>, <3 x i32> addrspace(3)* %in
  store <3 x i32> %ld, <3 x i32> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_load_v4i32:
; SICIVI: s_mov_b32 m0, -1
; GFX9-NOT: m0

; GCN: ds_read2_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset1:1{{$}}

define amdgpu_kernel void @local_load_v4i32(<4 x i32> addrspace(3)* %out, <4 x i32> addrspace(3)* %in) #0 {
entry:
  %ld = load <4 x i32>, <4 x i32> addrspace(3)* %in
  store <4 x i32> %ld, <4 x i32> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_load_v8i32:
; SICIVI: s_mov_b32 m0, -1
; GFX9-NOT: m0

; GCN-DAG: ds_read2_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset0:2 offset1:3{{$}}
; GCN-DAG: ds_read2_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset1:1{{$}}
define amdgpu_kernel void @local_load_v8i32(<8 x i32> addrspace(3)* %out, <8 x i32> addrspace(3)* %in) #0 {
entry:
  %ld = load <8 x i32>, <8 x i32> addrspace(3)* %in
  store <8 x i32> %ld, <8 x i32> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_load_v16i32:
; SICIVI: s_mov_b32 m0, -1
; GFX9-NOT: m0

; GCN-DAG: ds_read2_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset0:6 offset1:7{{$}}
; GCN-DAG: ds_read2_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset0:4 offset1:5{{$}}
; GCN-DAG: ds_read2_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset0:2 offset1:3{{$}}
; GCN-DAG: ds_read2_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset1:1{{$}}
; GCN-DAG: ds_write2_b64 v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}} offset0:6 offset1:7
; GCN-DAG: ds_write2_b64 v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}} offset0:4 offset1:5
; GCN-DAG: ds_write2_b64 v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}} offset0:2 offset1:3
; GCN-DAG: ds_write2_b64 v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}} offset1:1
define amdgpu_kernel void @local_load_v16i32(<16 x i32> addrspace(3)* %out, <16 x i32> addrspace(3)* %in) #0 {
entry:
  %ld = load <16 x i32>, <16 x i32> addrspace(3)* %in
  store <16 x i32> %ld, <16 x i32> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_i32_to_i64:
; SICIVI: s_mov_b32 m0, -1
; GFX9-NOT: m0

define amdgpu_kernel void @local_zextload_i32_to_i64(i64 addrspace(3)* %out, i32 addrspace(3)* %in) #0 {
  %ld = load i32, i32 addrspace(3)* %in
  %ext = zext i32 %ld to i64
  store i64 %ext, i64 addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_i32_to_i64:
; SICIVI: s_mov_b32 m0, -1
; GFX9-NOT: m0

define amdgpu_kernel void @local_sextload_i32_to_i64(i64 addrspace(3)* %out, i32 addrspace(3)* %in) #0 {
  %ld = load i32, i32 addrspace(3)* %in
  %ext = sext i32 %ld to i64
  store i64 %ext, i64 addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v1i32_to_v1i64:
; SICIVI: s_mov_b32 m0, -1
; GFX9-NOT: m0

define amdgpu_kernel void @local_zextload_v1i32_to_v1i64(<1 x i64> addrspace(3)* %out, <1 x i32> addrspace(3)* %in) #0 {
  %ld = load <1 x i32>, <1 x i32> addrspace(3)* %in
  %ext = zext <1 x i32> %ld to <1 x i64>
  store <1 x i64> %ext, <1 x i64> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v1i32_to_v1i64:
; SICIVI: s_mov_b32 m0, -1
; GFX9-NOT: m0

define amdgpu_kernel void @local_sextload_v1i32_to_v1i64(<1 x i64> addrspace(3)* %out, <1 x i32> addrspace(3)* %in) #0 {
  %ld = load <1 x i32>, <1 x i32> addrspace(3)* %in
  %ext = sext <1 x i32> %ld to <1 x i64>
  store <1 x i64> %ext, <1 x i64> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v2i32_to_v2i64:
; SICIVI: s_mov_b32 m0, -1
; GFX9-NOT: m0

define amdgpu_kernel void @local_zextload_v2i32_to_v2i64(<2 x i64> addrspace(3)* %out, <2 x i32> addrspace(3)* %in) #0 {
  %ld = load <2 x i32>, <2 x i32> addrspace(3)* %in
  %ext = zext <2 x i32> %ld to <2 x i64>
  store <2 x i64> %ext, <2 x i64> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v2i32_to_v2i64:
; SICIVI: s_mov_b32 m0, -1
; GFX9-NOT: m0

define amdgpu_kernel void @local_sextload_v2i32_to_v2i64(<2 x i64> addrspace(3)* %out, <2 x i32> addrspace(3)* %in) #0 {
  %ld = load <2 x i32>, <2 x i32> addrspace(3)* %in
  %ext = sext <2 x i32> %ld to <2 x i64>
  store <2 x i64> %ext, <2 x i64> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v4i32_to_v4i64:
; SICIVI: s_mov_b32 m0, -1
; GFX9-NOT: m0

define amdgpu_kernel void @local_zextload_v4i32_to_v4i64(<4 x i64> addrspace(3)* %out, <4 x i32> addrspace(3)* %in) #0 {
  %ld = load <4 x i32>, <4 x i32> addrspace(3)* %in
  %ext = zext <4 x i32> %ld to <4 x i64>
  store <4 x i64> %ext, <4 x i64> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v4i32_to_v4i64:
; SICIVI: s_mov_b32 m0, -1
; GFX9-NOT: m0

define amdgpu_kernel void @local_sextload_v4i32_to_v4i64(<4 x i64> addrspace(3)* %out, <4 x i32> addrspace(3)* %in) #0 {
  %ld = load <4 x i32>, <4 x i32> addrspace(3)* %in
  %ext = sext <4 x i32> %ld to <4 x i64>
  store <4 x i64> %ext, <4 x i64> addrspace(3)* %out
  ret void
}

; Tests if ds_read/write_b128 gets generated for the 16 byte aligned load.
; FUNC-LABEL: {{^}}local_v4i32_to_128:

; SI-NOT: ds_read_b128
; SI-NOT: ds_write_b128

; CIVI: ds_read_b128
; CIVI: ds_write_b128

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
define amdgpu_kernel void @local_v4i32_to_128(<4 x i32> addrspace(3)* %out, <4 x i32> addrspace(3)* %in) {
  %ld = load <4 x i32>, <4 x i32> addrspace(3)* %in, align 16
  store <4 x i32> %ld, <4 x i32> addrspace(3)* %out, align 16
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v8i32_to_v8i64:
; SICIVI: s_mov_b32 m0, -1
; GFX9-NOT: m0

define amdgpu_kernel void @local_zextload_v8i32_to_v8i64(<8 x i64> addrspace(3)* %out, <8 x i32> addrspace(3)* %in) #0 {
  %ld = load <8 x i32>, <8 x i32> addrspace(3)* %in
  %ext = zext <8 x i32> %ld to <8 x i64>
  store <8 x i64> %ext, <8 x i64> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v8i32_to_v8i64:
; SICIVI: s_mov_b32 m0, -1
; GFX9-NOT: m0

define amdgpu_kernel void @local_sextload_v8i32_to_v8i64(<8 x i64> addrspace(3)* %out, <8 x i32> addrspace(3)* %in) #0 {
  %ld = load <8 x i32>, <8 x i32> addrspace(3)* %in
  %ext = sext <8 x i32> %ld to <8 x i64>
  store <8 x i64> %ext, <8 x i64> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v16i32_to_v16i64:
; SICIVI: s_mov_b32 m0, -1
; GFX9-NOT: m0

define amdgpu_kernel void @local_sextload_v16i32_to_v16i64(<16 x i64> addrspace(3)* %out, <16 x i32> addrspace(3)* %in) #0 {
  %ld = load <16 x i32>, <16 x i32> addrspace(3)* %in
  %ext = sext <16 x i32> %ld to <16 x i64>
  store <16 x i64> %ext, <16 x i64> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v16i32_to_v16i64
; SICIVI: s_mov_b32 m0, -1
; GFX9-NOT: m0

define amdgpu_kernel void @local_zextload_v16i32_to_v16i64(<16 x i64> addrspace(3)* %out, <16 x i32> addrspace(3)* %in) #0 {
  %ld = load <16 x i32>, <16 x i32> addrspace(3)* %in
  %ext = zext <16 x i32> %ld to <16 x i64>
  store <16 x i64> %ext, <16 x i64> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_sextload_v32i32_to_v32i64:
; SICIVI: s_mov_b32 m0, -1
; GFX9-NOT: m0

define amdgpu_kernel void @local_sextload_v32i32_to_v32i64(<32 x i64> addrspace(3)* %out, <32 x i32> addrspace(3)* %in) #0 {
  %ld = load <32 x i32>, <32 x i32> addrspace(3)* %in
  %ext = sext <32 x i32> %ld to <32 x i64>
  store <32 x i64> %ext, <32 x i64> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_zextload_v32i32_to_v32i64:
; SICIVI: s_mov_b32 m0, -1
; GFX9-NOT: m0

define amdgpu_kernel void @local_zextload_v32i32_to_v32i64(<32 x i64> addrspace(3)* %out, <32 x i32> addrspace(3)* %in) #0 {
  %ld = load <32 x i32>, <32 x i32> addrspace(3)* %in
  %ext = zext <32 x i32> %ld to <32 x i64>
  store <32 x i64> %ext, <32 x i64> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_load_v32i32:
; SICIVI: s_mov_b32 m0, -1
; GFX9-NOT: m0
; GFX9-NOT: accvgpr

define amdgpu_kernel void @local_load_v32i32(<32 x i32> addrspace(3)* %out, <32 x i32> addrspace(3)* %in) #0 {
  %ld = load <32 x i32>, <32 x i32> addrspace(3)* %in
  store <32 x i32> %ld, <32 x i32> addrspace(3)* %out
  ret void
}

attributes #0 = { nounwind }
