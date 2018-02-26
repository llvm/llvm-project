; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SICIVI,FUNC %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SICIVI,FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SICIVI,FUNC %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX9,FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefixes=EG,FUNC %s

; FUNC-LABEL: {{^}}local_load_f64:
; SICIV: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_read_b64 [[VAL:v\[[0-9]+:[0-9]+\]]], v{{[0-9]+}}{{$}}
; GCN: ds_write_b64 v{{[0-9]+}}, [[VAL]]

; EG: LDS_READ_RET
; EG: LDS_READ_RET
define amdgpu_kernel void @local_load_f64(double addrspace(3)* %out, double addrspace(3)* %in) #0 {
  %ld = load double, double addrspace(3)* %in
  store double %ld, double addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_load_v2f64:
; SICIV: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_read2_b64

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
define amdgpu_kernel void @local_load_v2f64(<2 x double> addrspace(3)* %out, <2 x double> addrspace(3)* %in) #0 {
entry:
  %ld = load <2 x double>, <2 x double> addrspace(3)* %in
  store <2 x double> %ld, <2 x double> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_load_v3f64:
; SICIV: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-DAG: ds_read2_b64
; GCN-DAG: ds_read_b64

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
define amdgpu_kernel void @local_load_v3f64(<3 x double> addrspace(3)* %out, <3 x double> addrspace(3)* %in) #0 {
entry:
  %ld = load <3 x double>, <3 x double> addrspace(3)* %in
  store <3 x double> %ld, <3 x double> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_load_v4f64:
; SICIV: s_mov_b32 m0
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
define amdgpu_kernel void @local_load_v4f64(<4 x double> addrspace(3)* %out, <4 x double> addrspace(3)* %in) #0 {
entry:
  %ld = load <4 x double>, <4 x double> addrspace(3)* %in
  store <4 x double> %ld, <4 x double> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_load_v8f64:
; SICIV: s_mov_b32 m0
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
define amdgpu_kernel void @local_load_v8f64(<8 x double> addrspace(3)* %out, <8 x double> addrspace(3)* %in) #0 {
entry:
  %ld = load <8 x double>, <8 x double> addrspace(3)* %in
  store <8 x double> %ld, <8 x double> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_load_v16f64:
; SICIV: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_read2_b64
; GCN: ds_read2_b64
; GCN: ds_read2_b64
; GCN: ds_read2_b64
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
define amdgpu_kernel void @local_load_v16f64(<16 x double> addrspace(3)* %out, <16 x double> addrspace(3)* %in) #0 {
entry:
  %ld = load <16 x double>, <16 x double> addrspace(3)* %in
  store <16 x double> %ld, <16 x double> addrspace(3)* %out
  ret void
}

attributes #0 = { nounwind }
