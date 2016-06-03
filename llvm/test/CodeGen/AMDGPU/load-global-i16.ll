; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GCN-NOHSA -check-prefix=FUNC %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GCN-HSA -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GCN-NOHSA -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cayman < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; FIXME: r600 is broken because the bigger testcases spill and it's not implemented

; FUNC-LABEL: {{^}}global_load_i16:
; GCN-NOHSA: buffer_load_ushort v{{[0-9]+}}
; GCN-HSA: flat_load_ushort

; EG: VTX_READ_16 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0
define void @global_load_i16(i16 addrspace(1)* %out, i16 addrspace(1)* %in) {
entry:
  %ld = load i16, i16 addrspace(1)* %in
  store i16 %ld, i16 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v2i16:
; GCN-NOHSA: buffer_load_dword v
; GCN-HSA: flat_load_dword v

; EG: VTX_READ_32
define void @global_load_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %in) {
entry:
  %ld = load <2 x i16>, <2 x i16> addrspace(1)* %in
  store <2 x i16> %ld, <2 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v3i16:
; GCN-NOHSA: buffer_load_dwordx2 v
; GCN-HSA: flat_load_dwordx2 v

; EG-DAG: VTX_READ_32
; EG-DAG: VTX_READ_16
define void @global_load_v3i16(<3 x i16> addrspace(1)* %out, <3 x i16> addrspace(1)* %in) {
entry:
  %ld = load <3 x i16>, <3 x i16> addrspace(1)* %in
  store <3 x i16> %ld, <3 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v4i16:
; GCN-NOHSA: buffer_load_dwordx2
; GCN-HSA: flat_load_dwordx2

; EG: VTX_READ_64
define void @global_load_v4i16(<4 x i16> addrspace(1)* %out, <4 x i16> addrspace(1)* %in) {
entry:
  %ld = load <4 x i16>, <4 x i16> addrspace(1)* %in
  store <4 x i16> %ld, <4 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v8i16:
; GCN-NOHSA: buffer_load_dwordx4
; GCN-HSA: flat_load_dwordx4

; EG: VTX_READ_128
define void @global_load_v8i16(<8 x i16> addrspace(1)* %out, <8 x i16> addrspace(1)* %in) {
entry:
  %ld = load <8 x i16>, <8 x i16> addrspace(1)* %in
  store <8 x i16> %ld, <8 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v16i16:
; GCN-NOHSA: buffer_load_dwordx4
; GCN-NOHSA: buffer_load_dwordx4

; GCN-HSA: flat_load_dwordx4
; GCN-HSA: flat_load_dwordx4

; EG: VTX_READ_128
; EG: VTX_READ_128
define void @global_load_v16i16(<16 x i16> addrspace(1)* %out, <16 x i16> addrspace(1)* %in) {
entry:
  %ld = load <16 x i16>, <16 x i16> addrspace(1)* %in
  store <16 x i16> %ld, <16 x i16> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_i16_to_i32:
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_store_dword

; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_store_dword

; EG: VTX_READ_16 T{{[0-9]+\.X, T[0-9]+\.X}}
define void @global_zextload_i16_to_i32(i32 addrspace(1)* %out, i16 addrspace(1)* %in) #0 {
  %a = load i16, i16 addrspace(1)* %in
  %ext = zext i16 %a to i32
  store i32 %ext, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_i16_to_i32:
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_store_dword

; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_store_dword

; EG: VTX_READ_16 [[DST:T[0-9]\.[XYZW]]], [[DST]]
; EG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, [[DST]], 0.0, literal
; EG: 16
define void @global_sextload_i16_to_i32(i32 addrspace(1)* %out, i16 addrspace(1)* %in) #0 {
  %a = load i16, i16 addrspace(1)* %in
  %ext = sext i16 %a to i32
  store i32 %ext, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v1i16_to_v1i32:
; GCN-NOHSA: buffer_load_ushort
; GCN-HSA: flat_load_ushort
define void @global_zextload_v1i16_to_v1i32(<1 x i32> addrspace(1)* %out, <1 x i16> addrspace(1)* %in) #0 {
  %load = load <1 x i16>, <1 x i16> addrspace(1)* %in
  %ext = zext <1 x i16> %load to <1 x i32>
  store <1 x i32> %ext, <1 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v1i16_to_v1i32:
; GCN-NOHSA: buffer_load_sshort
; GCN-HSA: flat_load_sshort
define void @global_sextload_v1i16_to_v1i32(<1 x i32> addrspace(1)* %out, <1 x i16> addrspace(1)* %in) #0 {
  %load = load <1 x i16>, <1 x i16> addrspace(1)* %in
  %ext = sext <1 x i16> %load to <1 x i32>
  store <1 x i32> %ext, <1 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v2i16_to_v2i32:
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
define void @global_zextload_v2i16_to_v2i32(<2 x i32> addrspace(1)* %out, <2 x i16> addrspace(1)* %in) #0 {
  %load = load <2 x i16>, <2 x i16> addrspace(1)* %in
  %ext = zext <2 x i16> %load to <2 x i32>
  store <2 x i32> %ext, <2 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v2i16_to_v2i32:
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_load_sshort

; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_load_sshort

; EG-DAG: VTX_READ_16 [[DST_X:T[0-9]\.[XYZW]]], [[DST_X]]
; EG-DAG: VTX_READ_16 [[DST_Y:T[0-9]\.[XYZW]]], [[DST_Y]]
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, [[DST_X]], 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, [[DST_Y]], 0.0, literal
; EG-DAG: 16
; EG-DAG: 16
define void @global_sextload_v2i16_to_v2i32(<2 x i32> addrspace(1)* %out, <2 x i16> addrspace(1)* %in) #0 {
  %load = load <2 x i16>, <2 x i16> addrspace(1)* %in
  %ext = sext <2 x i16> %load to <2 x i32>
  store <2 x i32> %ext, <2 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_global_zextload_v3i16_to_v3i32:
; GCN-NOHSA: buffer_load_dwordx2
; GCN-HSA: flat_load_dwordx2
define void @global_global_zextload_v3i16_to_v3i32(<3 x i32> addrspace(1)* %out, <3 x i16> addrspace(1)* %in) {
entry:
  %ld = load <3 x i16>, <3 x i16> addrspace(1)* %in
  %ext = zext <3 x i16> %ld to <3 x i32>
  store <3 x i32> %ext, <3 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_global_sextload_v3i16_to_v3i32:
; GCN-NOHSA: buffer_load_dwordx2
; GCN-HSA: flat_load_dwordx2
define void @global_global_sextload_v3i16_to_v3i32(<3 x i32> addrspace(1)* %out, <3 x i16> addrspace(1)* %in) {
entry:
  %ld = load <3 x i16>, <3 x i16> addrspace(1)* %in
  %ext = sext <3 x i16> %ld to <3 x i32>
  store <3 x i32> %ext, <3 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_global_zextload_v4i16_to_v4i32:
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort

; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort

; EG: VTX_READ_16
; EG: VTX_READ_16
; EG: VTX_READ_16
; EG: VTX_READ_16
define void @global_global_zextload_v4i16_to_v4i32(<4 x i32> addrspace(1)* %out, <4 x i16> addrspace(1)* %in) #0 {
  %load = load <4 x i16>, <4 x i16> addrspace(1)* %in
  %ext = zext <4 x i16> %load to <4 x i32>
  store <4 x i32> %ext, <4 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v4i16_to_v4i32:
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_load_sshort

; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_load_sshort

; EG-DAG: VTX_READ_16 [[DST_X:T[0-9]\.[XYZW]]], [[DST_X]]
; EG-DAG: VTX_READ_16 [[DST_Y:T[0-9]\.[XYZW]]], [[DST_Y]]
; EG-DAG: VTX_READ_16 [[DST_Z:T[0-9]\.[XYZW]]], [[DST_Z]]
; EG-DAG: VTX_READ_16 [[DST_W:T[0-9]\.[XYZW]]], [[DST_W]]
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, [[DST_X]], 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, [[DST_Y]], 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, [[DST_Z]], 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, [[DST_W]], 0.0, literal
; EG-DAG: 16
; EG-DAG: 16
; EG-DAG: 16
; EG-DAG: 16
define void @global_sextload_v4i16_to_v4i32(<4 x i32> addrspace(1)* %out, <4 x i16> addrspace(1)* %in) #0 {
  %load = load <4 x i16>, <4 x i16> addrspace(1)* %in
  %ext = sext <4 x i16> %load to <4 x i32>
  store <4 x i32> %ext, <4 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v8i16_to_v8i32:
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort

; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
define void @global_zextload_v8i16_to_v8i32(<8 x i32> addrspace(1)* %out, <8 x i16> addrspace(1)* %in) #0 {
  %load = load <8 x i16>, <8 x i16> addrspace(1)* %in
  %ext = zext <8 x i16> %load to <8 x i32>
  store <8 x i32> %ext, <8 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v8i16_to_v8i32:
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_load_sshort

; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_load_sshort
define void @global_sextload_v8i16_to_v8i32(<8 x i32> addrspace(1)* %out, <8 x i16> addrspace(1)* %in) #0 {
  %load = load <8 x i16>, <8 x i16> addrspace(1)* %in
  %ext = sext <8 x i16> %load to <8 x i32>
  store <8 x i32> %ext, <8 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v16i16_to_v16i32:
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort

; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
define void @global_zextload_v16i16_to_v16i32(<16 x i32> addrspace(1)* %out, <16 x i16> addrspace(1)* %in) #0 {
  %load = load <16 x i16>, <16 x i16> addrspace(1)* %in
  %ext = zext <16 x i16> %load to <16 x i32>
  store <16 x i32> %ext, <16 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v16i16_to_v16i32:
define void @global_sextload_v16i16_to_v16i32(<16 x i32> addrspace(1)* %out, <16 x i16> addrspace(1)* %in) #0 {
  %load = load <16 x i16>, <16 x i16> addrspace(1)* %in
  %ext = sext <16 x i16> %load to <16 x i32>
  store <16 x i32> %ext, <16 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v32i16_to_v32i32:
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort

; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
define void @global_zextload_v32i16_to_v32i32(<32 x i32> addrspace(1)* %out, <32 x i16> addrspace(1)* %in) #0 {
  %load = load <32 x i16>, <32 x i16> addrspace(1)* %in
  %ext = zext <32 x i16> %load to <32 x i32>
  store <32 x i32> %ext, <32 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v32i16_to_v32i32:
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_load_sshort
; GCN-NOHSA: buffer_load_sshort

; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_load_sshort
; GCN-HSA: flat_load_sshort
define void @global_sextload_v32i16_to_v32i32(<32 x i32> addrspace(1)* %out, <32 x i16> addrspace(1)* %in) #0 {
  %load = load <32 x i16>, <32 x i16> addrspace(1)* %in
  %ext = sext <32 x i16> %load to <32 x i32>
  store <32 x i32> %ext, <32 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v64i16_to_v64i32:
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort
; GCN-NOHSA: buffer_load_ushort

; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
; GCN-HSA: flat_load_ushort
define void @global_zextload_v64i16_to_v64i32(<64 x i32> addrspace(1)* %out, <64 x i16> addrspace(1)* %in) #0 {
  %load = load <64 x i16>, <64 x i16> addrspace(1)* %in
  %ext = zext <64 x i16> %load to <64 x i32>
  store <64 x i32> %ext, <64 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v64i16_to_v64i32:
define void @global_sextload_v64i16_to_v64i32(<64 x i32> addrspace(1)* %out, <64 x i16> addrspace(1)* %in) #0 {
  %load = load <64 x i16>, <64 x i16> addrspace(1)* %in
  %ext = sext <64 x i16> %load to <64 x i32>
  store <64 x i32> %ext, <64 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_i16_to_i64:
; GCN-NOHSA-DAG: buffer_load_ushort v[[LO:[0-9]+]],
; GCN-HSA-DAG: flat_load_ushort v[[LO:[0-9]+]],
; GCN-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], 0{{$}}

; GCN-NOHSA: buffer_store_dwordx2 v{{\[}}[[LO]]:[[HI]]]
; GCN-HSA: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[}}[[LO]]:[[HI]]{{\]}}
define void @global_zextload_i16_to_i64(i64 addrspace(1)* %out, i16 addrspace(1)* %in) #0 {
  %a = load i16, i16 addrspace(1)* %in
  %ext = zext i16 %a to i64
  store i64 %ext, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_i16_to_i64:
; GCN-NOHSA-DAG: buffer_load_sshort v[[LO:[0-9]+]],
; GCN-HSA-DAG: flat_load_sshort v[[LO:[0-9]+]],
; GCN-DAG: v_ashrrev_i32_e32 v[[HI:[0-9]+]], 31, v[[LO]]

; GCN-NOHSA: buffer_store_dwordx2 v{{\[}}[[LO]]:[[HI]]]
; GCN-HSA: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[}}[[LO]]:[[HI]]{{\]}}
define void @global_sextload_i16_to_i64(i64 addrspace(1)* %out, i16 addrspace(1)* %in) #0 {
  %a = load i16, i16 addrspace(1)* %in
  %ext = sext i16 %a to i64
  store i64 %ext, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v1i16_to_v1i64:
define void @global_zextload_v1i16_to_v1i64(<1 x i64> addrspace(1)* %out, <1 x i16> addrspace(1)* %in) #0 {
  %load = load <1 x i16>, <1 x i16> addrspace(1)* %in
  %ext = zext <1 x i16> %load to <1 x i64>
  store <1 x i64> %ext, <1 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v1i16_to_v1i64:
define void @global_sextload_v1i16_to_v1i64(<1 x i64> addrspace(1)* %out, <1 x i16> addrspace(1)* %in) #0 {
  %load = load <1 x i16>, <1 x i16> addrspace(1)* %in
  %ext = sext <1 x i16> %load to <1 x i64>
  store <1 x i64> %ext, <1 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v2i16_to_v2i64:
define void @global_zextload_v2i16_to_v2i64(<2 x i64> addrspace(1)* %out, <2 x i16> addrspace(1)* %in) #0 {
  %load = load <2 x i16>, <2 x i16> addrspace(1)* %in
  %ext = zext <2 x i16> %load to <2 x i64>
  store <2 x i64> %ext, <2 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v2i16_to_v2i64:
define void @global_sextload_v2i16_to_v2i64(<2 x i64> addrspace(1)* %out, <2 x i16> addrspace(1)* %in) #0 {
  %load = load <2 x i16>, <2 x i16> addrspace(1)* %in
  %ext = sext <2 x i16> %load to <2 x i64>
  store <2 x i64> %ext, <2 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v4i16_to_v4i64:
define void @global_zextload_v4i16_to_v4i64(<4 x i64> addrspace(1)* %out, <4 x i16> addrspace(1)* %in) #0 {
  %load = load <4 x i16>, <4 x i16> addrspace(1)* %in
  %ext = zext <4 x i16> %load to <4 x i64>
  store <4 x i64> %ext, <4 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v4i16_to_v4i64:
define void @global_sextload_v4i16_to_v4i64(<4 x i64> addrspace(1)* %out, <4 x i16> addrspace(1)* %in) #0 {
  %load = load <4 x i16>, <4 x i16> addrspace(1)* %in
  %ext = sext <4 x i16> %load to <4 x i64>
  store <4 x i64> %ext, <4 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v8i16_to_v8i64:
define void @global_zextload_v8i16_to_v8i64(<8 x i64> addrspace(1)* %out, <8 x i16> addrspace(1)* %in) #0 {
  %load = load <8 x i16>, <8 x i16> addrspace(1)* %in
  %ext = zext <8 x i16> %load to <8 x i64>
  store <8 x i64> %ext, <8 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v8i16_to_v8i64:
define void @global_sextload_v8i16_to_v8i64(<8 x i64> addrspace(1)* %out, <8 x i16> addrspace(1)* %in) #0 {
  %load = load <8 x i16>, <8 x i16> addrspace(1)* %in
  %ext = sext <8 x i16> %load to <8 x i64>
  store <8 x i64> %ext, <8 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v16i16_to_v16i64:
define void @global_zextload_v16i16_to_v16i64(<16 x i64> addrspace(1)* %out, <16 x i16> addrspace(1)* %in) #0 {
  %load = load <16 x i16>, <16 x i16> addrspace(1)* %in
  %ext = zext <16 x i16> %load to <16 x i64>
  store <16 x i64> %ext, <16 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v16i16_to_v16i64:
define void @global_sextload_v16i16_to_v16i64(<16 x i64> addrspace(1)* %out, <16 x i16> addrspace(1)* %in) #0 {
  %load = load <16 x i16>, <16 x i16> addrspace(1)* %in
  %ext = sext <16 x i16> %load to <16 x i64>
  store <16 x i64> %ext, <16 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v32i16_to_v32i64:
define void @global_zextload_v32i16_to_v32i64(<32 x i64> addrspace(1)* %out, <32 x i16> addrspace(1)* %in) #0 {
  %load = load <32 x i16>, <32 x i16> addrspace(1)* %in
  %ext = zext <32 x i16> %load to <32 x i64>
  store <32 x i64> %ext, <32 x i64> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v32i16_to_v32i64:
define void @global_sextload_v32i16_to_v32i64(<32 x i64> addrspace(1)* %out, <32 x i16> addrspace(1)* %in) #0 {
  %load = load <32 x i16>, <32 x i16> addrspace(1)* %in
  %ext = sext <32 x i16> %load to <32 x i64>
  store <32 x i64> %ext, <32 x i64> addrspace(1)* %out
  ret void
}

; ; XFUNC-LABEL: {{^}}global_zextload_v64i16_to_v64i64:
; define void @global_zextload_v64i16_to_v64i64(<64 x i64> addrspace(1)* %out, <64 x i16> addrspace(1)* %in) #0 {
;   %load = load <64 x i16>, <64 x i16> addrspace(1)* %in
;   %ext = zext <64 x i16> %load to <64 x i64>
;   store <64 x i64> %ext, <64 x i64> addrspace(1)* %out
;   ret void
; }

; ; XFUNC-LABEL: {{^}}global_sextload_v64i16_to_v64i64:
; define void @global_sextload_v64i16_to_v64i64(<64 x i64> addrspace(1)* %out, <64 x i16> addrspace(1)* %in) #0 {
;   %load = load <64 x i16>, <64 x i16> addrspace(1)* %in
;   %ext = sext <64 x i16> %load to <64 x i64>
;   store <64 x i64> %ext, <64 x i64> addrspace(1)* %out
;   ret void
; }

attributes #0 = { nounwind }
