; RUN: llc -mtriple=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,FUNC %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=kaveri -mattr=-enable-ds128 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,FUNC %s
; RUN: llc -mtriple=amdgcn -mcpu=tonga -mattr=-enable-ds128 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,FUNC %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -mattr=-enable-ds128 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX9,FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefixes=EG,FUNC %s

; Testing for ds_read_b128
; RUN: llc -mtriple=amdgcn -mcpu=tonga -verify-machineinstrs -mattr=+enable-ds128 < %s | FileCheck -check-prefixes=CIVI,FUNC %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -verify-machineinstrs -mattr=+enable-ds128 < %s | FileCheck -check-prefixes=CIVI,FUNC %s

; FUNC-LABEL: {{^}}local_load_f64:
; SICIV: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_read_b64 [[VAL:v\[[0-9]+:[0-9]+\]]], v{{[0-9]+}}{{$}}
; GCN: ds_write_b64 v{{[0-9]+}}, [[VAL]]

; EG: LDS_READ_RET
; EG: LDS_READ_RET
define amdgpu_kernel void @local_load_f64(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
  %ld = load double, ptr addrspace(3) %in
  store double %ld, ptr addrspace(3) %out
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
define amdgpu_kernel void @local_load_v2f64(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
entry:
  %ld = load <2 x double>, ptr addrspace(3) %in
  store <2 x double> %ld, ptr addrspace(3) %out
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
define amdgpu_kernel void @local_load_v3f64(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
entry:
  %ld = load <3 x double>, ptr addrspace(3) %in
  store <3 x double> %ld, ptr addrspace(3) %out
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
define amdgpu_kernel void @local_load_v4f64(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
entry:
  %ld = load <4 x double>, ptr addrspace(3) %in
  store <4 x double> %ld, ptr addrspace(3) %out
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
define amdgpu_kernel void @local_load_v8f64(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
entry:
  %ld = load <8 x double>, ptr addrspace(3) %in
  store <8 x double> %ld, ptr addrspace(3) %out
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
define amdgpu_kernel void @local_load_v16f64(ptr addrspace(3) %out, ptr addrspace(3) %in) #0 {
entry:
  %ld = load <16 x double>, ptr addrspace(3) %in
  store <16 x double> %ld, ptr addrspace(3) %out
  ret void
}

; Tests if ds_read_b128 gets generated for the 16 byte aligned load.
; FUNC-LABEL: {{^}}local_load_v2f64_to_128:

; CIVI: ds_read_b128
; CIVI: ds_write_b128

; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
; EG: LDS_READ_RET
define amdgpu_kernel void @local_load_v2f64_to_128(ptr addrspace(3) %out, ptr addrspace(3) %in) {
entry:
  %ld = load <2 x double>, ptr addrspace(3) %in, align 16
  store <2 x double> %ld, ptr addrspace(3) %out, align 16
  ret void
}

attributes #0 = { nounwind }
