; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=FUNC %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}local_load_i64:
; GCN: ds_read_b64 [[VAL:v\[[0-9]+:[0-9]+\]]], v{{[0-9]+}}{{$}}
; GCN: ds_write_b64 v{{[0-9]+}}, [[VAL]]

; EG: LDS_READ_RET
; EG: LDS_READ_RET
define amdgpu_kernel void @local_load_i64(i64 addrspace(3)* %out, i64 addrspace(3)* %in) #0 {
  %ld = load i64, i64 addrspace(3)* %in
  store i64 %ld, i64 addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_load_v2i64:
; GCN: ds_read2_b64

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
define amdgpu_kernel void @local_load_v2i64(<2 x i64> addrspace(3)* %out, <2 x i64> addrspace(3)* %in) #0 {
entry:
  %ld = load <2 x i64>, <2 x i64> addrspace(3)* %in
  store <2 x i64> %ld, <2 x i64> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_load_v3i64:
; GCN-DAG: ds_read2_b64
; GCN-DAG: ds_read_b64

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
define amdgpu_kernel void @local_load_v3i64(<3 x i64> addrspace(3)* %out, <3 x i64> addrspace(3)* %in) #0 {
entry:
  %ld = load <3 x i64>, <3 x i64> addrspace(3)* %in
  store <3 x i64> %ld, <3 x i64> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_load_v4i64:
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
define amdgpu_kernel void @local_load_v4i64(<4 x i64> addrspace(3)* %out, <4 x i64> addrspace(3)* %in) #0 {
entry:
  %ld = load <4 x i64>, <4 x i64> addrspace(3)* %in
  store <4 x i64> %ld, <4 x i64> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_load_v8i64:
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
define amdgpu_kernel void @local_load_v8i64(<8 x i64> addrspace(3)* %out, <8 x i64> addrspace(3)* %in) #0 {
entry:
  %ld = load <8 x i64>, <8 x i64> addrspace(3)* %in
  store <8 x i64> %ld, <8 x i64> addrspace(3)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_load_v16i64:
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
define amdgpu_kernel void @local_load_v16i64(<16 x i64> addrspace(3)* %out, <16 x i64> addrspace(3)* %in) #0 {
entry:
  %ld = load <16 x i64>, <16 x i64> addrspace(3)* %in
  store <16 x i64> %ld, <16 x i64> addrspace(3)* %out
  ret void
}

attributes #0 = { nounwind }
