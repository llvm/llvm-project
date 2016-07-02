; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GCN-NOHSA -check-prefix=FUNC %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GCN-HSA -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GCN-NOHSA -check-prefix=FUNC %s

; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cayman < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}global_load_i64:
; GCN-NOHSA: buffer_load_dwordx2 [[VAL:v\[[0-9]+:[0-9]+\]]]
; GCN-NOHSA: buffer_store_dwordx2 [[VAL]]

; GCN-HSA: flat_load_dwordx2 [[VAL:v\[[0-9]+:[0-9]+\]]]
; GCN-HSA: flat_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, [[VAL]]

; EG: VTX_READ_64
define void @global_load_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %in) #0 {
  %ld = load i64, i64 addrspace(1)* %in
  store i64 %ld, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v2i64:
; GCN-NOHSA: buffer_load_dwordx4
; GCN-HSA: flat_load_dwordx4

; EG: VTX_READ_128
define void @global_load_v2i64(<2 x i64> addrspace(1)* %out, <2 x i64> addrspace(1)* %in) #0 {
entry:
  %ld = load <2 x i64>, <2 x i64> addrspace(1)* %in
  store <2 x i64> %ld, <2 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v3i64:
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: buffer_load_dwordx4

; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4

; EG: VTX_READ_128
; EG: VTX_READ_128
define void @global_load_v3i64(<3 x i64> addrspace(1)* %out, <3 x i64> addrspace(1)* %in) #0 {
entry:
  %ld = load <3 x i64>, <3 x i64> addrspace(1)* %in
  store <3 x i64> %ld, <3 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v4i64:
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: buffer_load_dwordx4

; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4

; EG: VTX_READ_128
; EG: VTX_READ_128
define void @global_load_v4i64(<4 x i64> addrspace(1)* %out, <4 x i64> addrspace(1)* %in) #0 {
entry:
  %ld = load <4 x i64>, <4 x i64> addrspace(1)* %in
  store <4 x i64> %ld, <4 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v8i64:
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: buffer_load_dwordx4

; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4

; EG: VTX_READ_128
; EG: VTX_READ_128
; EG: VTX_READ_128
; EG: VTX_READ_128
define void @global_load_v8i64(<8 x i64> addrspace(1)* %out, <8 x i64> addrspace(1)* %in) #0 {
entry:
  %ld = load <8 x i64>, <8 x i64> addrspace(1)* %in
  store <8 x i64> %ld, <8 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v16i64:
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: buffer_load_dwordx4

; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4

; EG: VTX_READ_128
; EG: VTX_READ_128
; EG: VTX_READ_128
; EG: VTX_READ_128
; EG: VTX_READ_128
; EG: VTX_READ_128
; EG: VTX_READ_128
; EG: VTX_READ_128
define void @global_load_v16i64(<16 x i64> addrspace(1)* %out, <16 x i64> addrspace(1)* %in) #0 {
entry:
  %ld = load <16 x i64>, <16 x i64> addrspace(1)* %in
  store <16 x i64> %ld, <16 x i64> addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
