; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -verify-machineinstrs< %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs< %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; XUN: llc -march=r600 -mcpu=cypress < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s
; FIXME: cypress is broken because the bigger testcases spill and it's not implemented

; FUNC-LABEL: {{^}}zextload_global_i16_to_i32:
; SI: buffer_load_ushort
; SI: buffer_store_dword
; SI: s_endpgm
define amdgpu_kernel void @zextload_global_i16_to_i32(ptr addrspace(1) %out, ptr addrspace(1) %in) nounwind {
  %a = load i16, ptr addrspace(1) %in
  %ext = zext i16 %a to i32
  store i32 %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}sextload_global_i16_to_i32:
; SI: buffer_load_sshort
; SI: buffer_store_dword
; SI: s_endpgm
define amdgpu_kernel void @sextload_global_i16_to_i32(ptr addrspace(1) %out, ptr addrspace(1) %in) nounwind {
  %a = load i16, ptr addrspace(1) %in
  %ext = sext i16 %a to i32
  store i32 %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}zextload_global_v1i16_to_v1i32:
; SI: buffer_load_ushort
; SI: s_endpgm
define amdgpu_kernel void @zextload_global_v1i16_to_v1i32(ptr addrspace(1) %out, ptr addrspace(1) nocapture %in) nounwind {
  %load = load <1 x i16>, ptr addrspace(1) %in
  %ext = zext <1 x i16> %load to <1 x i32>
  store <1 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}sextload_global_v1i16_to_v1i32:
; SI: buffer_load_sshort
; SI: s_endpgm
define amdgpu_kernel void @sextload_global_v1i16_to_v1i32(ptr addrspace(1) %out, ptr addrspace(1) nocapture %in) nounwind {
  %load = load <1 x i16>, ptr addrspace(1) %in
  %ext = sext <1 x i16> %load to <1 x i32>
  store <1 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}zextload_global_v2i16_to_v2i32:
; SI: s_endpgm
define amdgpu_kernel void @zextload_global_v2i16_to_v2i32(ptr addrspace(1) %out, ptr addrspace(1) nocapture %in) nounwind {
  %load = load <2 x i16>, ptr addrspace(1) %in
  %ext = zext <2 x i16> %load to <2 x i32>
  store <2 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}sextload_global_v2i16_to_v2i32:
; SI: s_endpgm
define amdgpu_kernel void @sextload_global_v2i16_to_v2i32(ptr addrspace(1) %out, ptr addrspace(1) nocapture %in) nounwind {
  %load = load <2 x i16>, ptr addrspace(1) %in
  %ext = sext <2 x i16> %load to <2 x i32>
  store <2 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}zextload_global_v4i16_to_v4i32:
; SI: s_endpgm
define amdgpu_kernel void @zextload_global_v4i16_to_v4i32(ptr addrspace(1) %out, ptr addrspace(1) nocapture %in) nounwind {
  %load = load <4 x i16>, ptr addrspace(1) %in
  %ext = zext <4 x i16> %load to <4 x i32>
  store <4 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}sextload_global_v4i16_to_v4i32:
; SI: s_endpgm
define amdgpu_kernel void @sextload_global_v4i16_to_v4i32(ptr addrspace(1) %out, ptr addrspace(1) nocapture %in) nounwind {
  %load = load <4 x i16>, ptr addrspace(1) %in
  %ext = sext <4 x i16> %load to <4 x i32>
  store <4 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}zextload_global_v8i16_to_v8i32:
; SI: s_endpgm
define amdgpu_kernel void @zextload_global_v8i16_to_v8i32(ptr addrspace(1) %out, ptr addrspace(1) nocapture %in) nounwind {
  %load = load <8 x i16>, ptr addrspace(1) %in
  %ext = zext <8 x i16> %load to <8 x i32>
  store <8 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}sextload_global_v8i16_to_v8i32:
; SI: s_endpgm
define amdgpu_kernel void @sextload_global_v8i16_to_v8i32(ptr addrspace(1) %out, ptr addrspace(1) nocapture %in) nounwind {
  %load = load <8 x i16>, ptr addrspace(1) %in
  %ext = sext <8 x i16> %load to <8 x i32>
  store <8 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}zextload_global_v16i16_to_v16i32:
; SI: s_endpgm
define amdgpu_kernel void @zextload_global_v16i16_to_v16i32(ptr addrspace(1) %out, ptr addrspace(1) nocapture %in) nounwind {
  %load = load <16 x i16>, ptr addrspace(1) %in
  %ext = zext <16 x i16> %load to <16 x i32>
  store <16 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}sextload_global_v16i16_to_v16i32:
; SI: s_endpgm
define amdgpu_kernel void @sextload_global_v16i16_to_v16i32(ptr addrspace(1) %out, ptr addrspace(1) nocapture %in) nounwind {
  %load = load <16 x i16>, ptr addrspace(1) %in
  %ext = sext <16 x i16> %load to <16 x i32>
  store <16 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}zextload_global_v32i16_to_v32i32:
; SI: s_endpgm
define amdgpu_kernel void @zextload_global_v32i16_to_v32i32(ptr addrspace(1) %out, ptr addrspace(1) nocapture %in) nounwind {
  %load = load <32 x i16>, ptr addrspace(1) %in
  %ext = zext <32 x i16> %load to <32 x i32>
  store <32 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}sextload_global_v32i16_to_v32i32:
; SI: s_endpgm
define amdgpu_kernel void @sextload_global_v32i16_to_v32i32(ptr addrspace(1) %out, ptr addrspace(1) nocapture %in) nounwind {
  %load = load <32 x i16>, ptr addrspace(1) %in
  %ext = sext <32 x i16> %load to <32 x i32>
  store <32 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}zextload_global_v64i16_to_v64i32:
; SI: s_endpgm
define amdgpu_kernel void @zextload_global_v64i16_to_v64i32(ptr addrspace(1) %out, ptr addrspace(1) nocapture %in) nounwind {
  %load = load <64 x i16>, ptr addrspace(1) %in
  %ext = zext <64 x i16> %load to <64 x i32>
  store <64 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}sextload_global_v64i16_to_v64i32:
; SI: s_endpgm
define amdgpu_kernel void @sextload_global_v64i16_to_v64i32(ptr addrspace(1) %out, ptr addrspace(1) nocapture %in) nounwind {
  %load = load <64 x i16>, ptr addrspace(1) %in
  %ext = sext <64 x i16> %load to <64 x i32>
  store <64 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}zextload_global_i16_to_i64:
; SI-DAG: buffer_load_ushort v[[LO:[0-9]+]],
; SI-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], 0{{$}}
; SI: buffer_store_dwordx2 v[[[LO]]:[[HI]]]
define amdgpu_kernel void @zextload_global_i16_to_i64(ptr addrspace(1) %out, ptr addrspace(1) %in) nounwind {
  %a = load i16, ptr addrspace(1) %in
  %ext = zext i16 %a to i64
  store i64 %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}sextload_global_i16_to_i64:
; VI: buffer_load_ushort [[LOAD:v[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0
; VI: v_ashrrev_i32_e32 v{{[0-9]+}}, 31, [[LOAD]]
; VI: buffer_store_dwordx2 v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], 0
define amdgpu_kernel void @sextload_global_i16_to_i64(ptr addrspace(1) %out, ptr addrspace(1) %in) nounwind {
  %a = load i16, ptr addrspace(1) %in
  %ext = sext i16 %a to i64
  store i64 %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}zextload_global_v1i16_to_v1i64:
; SI: s_endpgm
define amdgpu_kernel void @zextload_global_v1i16_to_v1i64(ptr addrspace(1) %out, ptr addrspace(1) nocapture %in) nounwind {
  %load = load <1 x i16>, ptr addrspace(1) %in
  %ext = zext <1 x i16> %load to <1 x i64>
  store <1 x i64> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}sextload_global_v1i16_to_v1i64:
; SI: s_endpgm
define amdgpu_kernel void @sextload_global_v1i16_to_v1i64(ptr addrspace(1) %out, ptr addrspace(1) nocapture %in) nounwind {
  %load = load <1 x i16>, ptr addrspace(1) %in
  %ext = sext <1 x i16> %load to <1 x i64>
  store <1 x i64> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}zextload_global_v2i16_to_v2i64:
; SI: s_endpgm
define amdgpu_kernel void @zextload_global_v2i16_to_v2i64(ptr addrspace(1) %out, ptr addrspace(1) nocapture %in) nounwind {
  %load = load <2 x i16>, ptr addrspace(1) %in
  %ext = zext <2 x i16> %load to <2 x i64>
  store <2 x i64> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}sextload_global_v2i16_to_v2i64:
; SI: s_endpgm
define amdgpu_kernel void @sextload_global_v2i16_to_v2i64(ptr addrspace(1) %out, ptr addrspace(1) nocapture %in) nounwind {
  %load = load <2 x i16>, ptr addrspace(1) %in
  %ext = sext <2 x i16> %load to <2 x i64>
  store <2 x i64> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}zextload_global_v4i16_to_v4i64:
; SI: s_endpgm
define amdgpu_kernel void @zextload_global_v4i16_to_v4i64(ptr addrspace(1) %out, ptr addrspace(1) nocapture %in) nounwind {
  %load = load <4 x i16>, ptr addrspace(1) %in
  %ext = zext <4 x i16> %load to <4 x i64>
  store <4 x i64> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}sextload_global_v4i16_to_v4i64:
; SI: s_endpgm
define amdgpu_kernel void @sextload_global_v4i16_to_v4i64(ptr addrspace(1) %out, ptr addrspace(1) nocapture %in) nounwind {
  %load = load <4 x i16>, ptr addrspace(1) %in
  %ext = sext <4 x i16> %load to <4 x i64>
  store <4 x i64> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}zextload_global_v8i16_to_v8i64:
; SI: s_endpgm
define amdgpu_kernel void @zextload_global_v8i16_to_v8i64(ptr addrspace(1) %out, ptr addrspace(1) nocapture %in) nounwind {
  %load = load <8 x i16>, ptr addrspace(1) %in
  %ext = zext <8 x i16> %load to <8 x i64>
  store <8 x i64> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}sextload_global_v8i16_to_v8i64:
; SI: s_endpgm
define amdgpu_kernel void @sextload_global_v8i16_to_v8i64(ptr addrspace(1) %out, ptr addrspace(1) nocapture %in) nounwind {
  %load = load <8 x i16>, ptr addrspace(1) %in
  %ext = sext <8 x i16> %load to <8 x i64>
  store <8 x i64> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}zextload_global_v16i16_to_v16i64:
; SI: s_endpgm
define amdgpu_kernel void @zextload_global_v16i16_to_v16i64(ptr addrspace(1) %out, ptr addrspace(1) nocapture %in) nounwind {
  %load = load <16 x i16>, ptr addrspace(1) %in
  %ext = zext <16 x i16> %load to <16 x i64>
  store <16 x i64> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}sextload_global_v16i16_to_v16i64:
; SI: s_endpgm
define amdgpu_kernel void @sextload_global_v16i16_to_v16i64(ptr addrspace(1) %out, ptr addrspace(1) nocapture %in) nounwind {
  %load = load <16 x i16>, ptr addrspace(1) %in
  %ext = sext <16 x i16> %load to <16 x i64>
  store <16 x i64> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}zextload_global_v32i16_to_v32i64:
; SI: s_endpgm
define amdgpu_kernel void @zextload_global_v32i16_to_v32i64(ptr addrspace(1) %out, ptr addrspace(1) nocapture %in) nounwind {
  %load = load <32 x i16>, ptr addrspace(1) %in
  %ext = zext <32 x i16> %load to <32 x i64>
  store <32 x i64> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}sextload_global_v32i16_to_v32i64:
; SI: s_endpgm
define amdgpu_kernel void @sextload_global_v32i16_to_v32i64(ptr addrspace(1) %out, ptr addrspace(1) nocapture %in) nounwind {
  %load = load <32 x i16>, ptr addrspace(1) %in
  %ext = sext <32 x i16> %load to <32 x i64>
  store <32 x i64> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}zextload_global_v64i16_to_v64i64:
; SI: s_endpgm
define amdgpu_kernel void @zextload_global_v64i16_to_v64i64(ptr addrspace(1) %out, ptr addrspace(1) nocapture %in) nounwind {
  %load = load <64 x i16>, ptr addrspace(1) %in
  %ext = zext <64 x i16> %load to <64 x i64>
  store <64 x i64> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}sextload_global_v64i16_to_v64i64:
; SI: s_endpgm
define amdgpu_kernel void @sextload_global_v64i16_to_v64i64(ptr addrspace(1) %out, ptr addrspace(1) nocapture %in) nounwind {
  %load = load <64 x i16>, ptr addrspace(1) %in
  %ext = sext <64 x i16> %load to <64 x i64>
  store <64 x i64> %ext, ptr addrspace(1) %out
  ret void
}
