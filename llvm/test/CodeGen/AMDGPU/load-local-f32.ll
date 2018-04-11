; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SICIVI,FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SICIVI,FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefixes=EG,FUNC %s

; Testing for ds_read/write_128
; RUN: llc -march=amdgcn -mcpu=tahiti -mattr=+enable-ds128 < %s | FileCheck -check-prefixes=SI,FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=+enable-ds128 < %s | FileCheck -check-prefixes=CIVI,FUNC %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -mattr=+enable-ds128 < %s | FileCheck -check-prefixes=CIVI,FUNC %s

; FUNC-LABEL: {{^}}load_f32_local:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0
; GCN: ds_read_b32

; EG: LDS_READ_RET
define amdgpu_kernel void @load_f32_local(float addrspace(1)* %out, float addrspace(3)* %in) #0 {
entry:
  %tmp0 = load float, float addrspace(3)* %in
  store float %tmp0, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}load_v2f32_local:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_read_b64

; EG: LDS_READ_RET
; EG: LDS_READ_RET
define amdgpu_kernel void @load_v2f32_local(<2 x float> addrspace(1)* %out, <2 x float> addrspace(3)* %in) #0 {
entry:
  %tmp0 = load <2 x float>, <2 x float> addrspace(3)* %in
  store <2 x float> %tmp0, <2 x float> addrspace(1)* %out
  ret void
}

; FIXME: should this do a read2_b64?
; FUNC-LABEL: {{^}}local_load_v3f32:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-DAG: ds_read_b32 v{{[0-9]+}}, v{{[0-9]+}} offset:8
; GCN-DAG: ds_read_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+$}}
; GCN: s_waitcnt
; GCN-DAG: ds_write_b64
; GCN-DAG: ds_write_b32 v{{[0-9]+}}, v{{[0-9]+}} offset:8{{$}}

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
define amdgpu_kernel void @local_load_v3f32(<3 x float> addrspace(3)* %out, <3 x float> addrspace(3)* %in) #0 {
entry:
  %tmp0 = load <3 x float>, <3 x float> addrspace(3)* %in
  store <3 x float> %tmp0, <3 x float> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_load_v4f32:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_read2_b64

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
define amdgpu_kernel void @local_load_v4f32(<4 x float> addrspace(3)* %out, <4 x float> addrspace(3)* %in) #0 {
entry:
  %tmp0 = load <4 x float>, <4 x float> addrspace(3)* %in
  store <4 x float> %tmp0, <4 x float> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_load_v8f32:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_read2_b64
; GCN: ds_read2_b64

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
define amdgpu_kernel void @local_load_v8f32(<8 x float> addrspace(3)* %out, <8 x float> addrspace(3)* %in) #0 {
entry:
  %tmp0 = load <8 x float>, <8 x float> addrspace(3)* %in
  store <8 x float> %tmp0, <8 x float> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_load_v16f32:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_read2_b64
; GCN: ds_read2_b64
; GCN: ds_read2_b64
; GCN: ds_read2_b64

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
define amdgpu_kernel void @local_load_v16f32(<16 x float> addrspace(3)* %out, <16 x float> addrspace(3)* %in) #0 {
entry:
  %tmp0 = load <16 x float>, <16 x float> addrspace(3)* %in
  store <16 x float> %tmp0, <16 x float> addrspace(3)* %out
  ret void
}

; Tests if ds_read/write_b128 gets generated for the 16 byte aligned load.
; FUNC-LABEL: {{^}}local_v4f32_to_128:

; SI-NOT: ds_read_b128
; SI-NOT: ds_write_b128

; CIVI: ds_read_b128
; CIVI: ds_write_b128

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
define amdgpu_kernel void @local_v4f32_to_128(<4 x float> addrspace(3)* %out, <4 x float> addrspace(3)* %in) {
  %ld = load <4 x float>, <4 x float> addrspace(3)* %in, align 16
  store <4 x float> %ld, <4 x float> addrspace(3)* %out, align 16
  ret void
}

attributes #0 = { nounwind }
