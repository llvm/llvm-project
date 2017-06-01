; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GCN-NOHSA -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GCN-HSA -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GCN-NOHSA -check-prefix=VI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s


; FUNC-LABEL: {{^}}constant_load_i64:
; GCN: s_load_dwordx2 {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0x0{{$}}
; EG: VTX_READ_64
define void @constant_load_i64(i64 addrspace(1)* %out, i64 addrspace(2)* %in) #0 {
  %ld = load i64, i64 addrspace(2)* %in
  store i64 %ld, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_load_v2i64:
; GCN: s_load_dwordx4

; EG: VTX_READ_128
define void @constant_load_v2i64(<2 x i64> addrspace(1)* %out, <2 x i64> addrspace(2)* %in) #0 {
entry:
  %ld = load <2 x i64>, <2 x i64> addrspace(2)* %in
  store <2 x i64> %ld, <2 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_load_v3i64:
; GCN: s_load_dwordx8 {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0x0{{$}}

; EG-DAG: VTX_READ_128
; EG-DAG: VTX_READ_128
define void @constant_load_v3i64(<3 x i64> addrspace(1)* %out, <3 x i64> addrspace(2)* %in) #0 {
entry:
  %ld = load <3 x i64>, <3 x i64> addrspace(2)* %in
  store <3 x i64> %ld, <3 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_load_v4i64
; GCN: s_load_dwordx8

; EG: VTX_READ_128
; EG: VTX_READ_128
define void @constant_load_v4i64(<4 x i64> addrspace(1)* %out, <4 x i64> addrspace(2)* %in) #0 {
entry:
  %ld = load <4 x i64>, <4 x i64> addrspace(2)* %in
  store <4 x i64> %ld, <4 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_load_v8i64:
; GCN: s_load_dwordx16

; EG: VTX_READ_128
; EG: VTX_READ_128
; EG: VTX_READ_128
; EG: VTX_READ_128
define void @constant_load_v8i64(<8 x i64> addrspace(1)* %out, <8 x i64> addrspace(2)* %in) #0 {
entry:
  %ld = load <8 x i64>, <8 x i64> addrspace(2)* %in
  store <8 x i64> %ld, <8 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}constant_load_v16i64:
; GCN: s_load_dwordx16
; GCN: s_load_dwordx16

; EG: VTX_READ_128
; EG: VTX_READ_128
; EG: VTX_READ_128
; EG: VTX_READ_128
; EG: VTX_READ_128
; EG: VTX_READ_128
; EG: VTX_READ_128
; EG: VTX_READ_128
define void @constant_load_v16i64(<16 x i64> addrspace(1)* %out, <16 x i64> addrspace(2)* %in) #0 {
entry:
  %ld = load <16 x i64>, <16 x i64> addrspace(2)* %in
  store <16 x i64> %ld, <16 x i64> addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
