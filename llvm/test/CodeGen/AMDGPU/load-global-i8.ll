; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -verify-machineinstrs < %s | FileCheck -allow-deprecated-dag-overlap -check-prefixes=GCN,GCN-NOHSA,SI,FUNC %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -mtriple=amdgcn--amdhsa -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -allow-deprecated-dag-overlap -check-prefixes=GCN,GCN-HSA,SI,FUNC %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -allow-deprecated-dag-overlap -check-prefixes=GCN,GCN-NOHSA,VI,FUNC %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=r600 -mcpu=redwood -verify-machineinstrs < %s | FileCheck -allow-deprecated-dag-overlap -check-prefix=EG -check-prefix=FUNC %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=r600 -mcpu=cayman -verify-machineinstrs < %s | FileCheck -allow-deprecated-dag-overlap -check-prefix=EG -check-prefix=FUNC %s


; FUNC-LABEL: {{^}}global_load_i8:
; GCN-NOHSA: buffer_load_ubyte v{{[0-9]+}}
; GCN-HSA: flat_load_ubyte

; EG: VTX_READ_8 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0, #1
; TODO: NOT AND
define amdgpu_kernel void @global_load_i8(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
entry:
  %ld = load i8, ptr addrspace(1) %in
  store i8 %ld, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v2i8:
; GCN-NOHSA: buffer_load_ushort v
; GCN-HSA: flat_load_ushort v

; EG: VTX_READ_16 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0, #1
define amdgpu_kernel void @global_load_v2i8(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
entry:
  %ld = load <2 x i8>, ptr addrspace(1) %in
  store <2 x i8> %ld, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v3i8:
; GCN-NOHSA: buffer_load_dword v
; GCN-HSA: flat_load_dword v

; EG: VTX_READ_32 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0, #1
define amdgpu_kernel void @global_load_v3i8(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
entry:
  %ld = load <3 x i8>, ptr addrspace(1) %in
  store <3 x i8> %ld, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v4i8:
; GCN-NOHSA: buffer_load_dword v
; GCN-HSA: flat_load_dword v

; EG: VTX_READ_32 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0, #1
define amdgpu_kernel void @global_load_v4i8(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
entry:
  %ld = load <4 x i8>, ptr addrspace(1) %in
  store <4 x i8> %ld, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v8i8:
; GCN-NOHSA: buffer_load_dwordx2
; GCN-HSA: flat_load_dwordx2

; EG: VTX_READ_64 T{{[0-9]+}}.XY, T{{[0-9]+}}.X, 0, #1
define amdgpu_kernel void @global_load_v8i8(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
entry:
  %ld = load <8 x i8>, ptr addrspace(1) %in
  store <8 x i8> %ld, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_load_v16i8:
; GCN-NOHSA: buffer_load_dwordx4

; GCN-HSA: flat_load_dwordx4

; EG: VTX_READ_128 T{{[0-9]+}}.XYZW, T{{[0-9]+}}.X, 0, #1
define amdgpu_kernel void @global_load_v16i8(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
entry:
  %ld = load <16 x i8>, ptr addrspace(1) %in
  store <16 x i8> %ld, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_i8_to_i32:
; GCN-NOHSA: buffer_load_ubyte v{{[0-9]+}},
; GCN-HSA: flat_load_ubyte

; EG: VTX_READ_8 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0, #1
define amdgpu_kernel void @global_zextload_i8_to_i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %a = load i8, ptr addrspace(1) %in
  %ext = zext i8 %a to i32
  store i32 %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_i8_to_i32:
; GCN-NOHSA: buffer_load_sbyte
; GCN-HSA: flat_load_sbyte

; EG: VTX_READ_8 [[DST:T[0-9]\.[XYZW]]], [[DST]], 0, #1
; EG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, [[DST]], 0.0, literal
; EG: 8
define amdgpu_kernel void @global_sextload_i8_to_i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %ld = load i8, ptr addrspace(1) %in
  %ext = sext i8 %ld to i32
  store i32 %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v1i8_to_v1i32:

; EG: VTX_READ_8 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0, #1
define amdgpu_kernel void @global_zextload_v1i8_to_v1i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <1 x i8>, ptr addrspace(1) %in
  %ext = zext <1 x i8> %load to <1 x i32>
  store <1 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v1i8_to_v1i32:

; EG: VTX_READ_8 [[DST:T[0-9]\.[XYZW]]], [[DST]], 0, #1
; EG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, [[DST]], 0.0, literal
; EG: 8
define amdgpu_kernel void @global_sextload_v1i8_to_v1i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <1 x i8>, ptr addrspace(1) %in
  %ext = sext <1 x i8> %load to <1 x i32>
  store <1 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v2i8_to_v2i32:
; GCN-NOHSA: buffer_load_ushort
; GCN-HSA: flat_load_ushort

; EG: VTX_READ_16 [[DST:T[0-9]+\.X]], T{{[0-9]+}}.X, 0, #1
; TODO: These should use DST, but for some there are redundant MOVs
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, literal
; EG-DAG: 8
define amdgpu_kernel void @global_zextload_v2i8_to_v2i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <2 x i8>, ptr addrspace(1) %in
  %ext = zext <2 x i8> %load to <2 x i32>
  store <2 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v2i8_to_v2i32:
; GCN-NOHSA: buffer_load_ushort
; GCN-HSA: flat_load_ushort

; EG: VTX_READ_16 [[DST:T[0-9]+\.X]], T{{[0-9]+}}.X, 0, #1
; TODO: These should use DST, but for some there are redundant MOVs
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: 8
; EG-DAG: 8
define amdgpu_kernel void @global_sextload_v2i8_to_v2i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <2 x i8>, ptr addrspace(1) %in
  %ext = sext <2 x i8> %load to <2 x i32>
  store <2 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v3i8_to_v3i32:
; GCN-NOHSA: buffer_load_dword v
; GCN-HSA: flat_load_dword v

; SI-DAG: v_bfe_u32 v{{[0-9]+}}, v{{[0-9]+}}, 8, 8
; VI-DAG: v_lshrrev_b16_e32 v{{[0-9]+}}, 8, v{{[0-9]+}}
; GCN-DAG: v_bfe_u32 v{{[0-9]+}}, v{{[0-9]+}}, 16, 8
; GCN-DAG: v_and_b32_e32 v{{[0-9]+}}, 0xff,

; EG: VTX_READ_32 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0, #1
; TODO: These should use DST, but for some there are redundant MOVs
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: 8
; EG-DAG: 8
define amdgpu_kernel void @global_zextload_v3i8_to_v3i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
entry:
  %ld = load <3 x i8>, ptr addrspace(1) %in
  %ext = zext <3 x i8> %ld to <3 x i32>
  store <3 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v3i8_to_v3i32:
; GCN-NOHSA: buffer_load_dword v
; GCN-HSA: flat_load_dword v

;FIXME: Need to optimize this sequence to avoid extra shift on VI.

; t23: i16 = truncate t18
; t49: i16 = srl t23, Constant:i32<8>
; t57: i32 = any_extend t49
; t58: i32 = sign_extend_inreg t57, ValueType:ch:i8

; SI-DAG: v_bfe_i32 v{{[0-9]+}}, v{{[0-9]+}}, 8, 8
; VI-DAG: v_lshrrev_b16_e32 [[SHIFT:v[0-9]+]], 8, v{{[0-9]+}}
; VI-DAG: v_bfe_i32 v{{[0-9]+}}, [[SHIFT]], 0, 8
; GCN-DAG: v_bfe_i32 v{{[0-9]+}}, v{{[0-9]+}}, 0, 8
; GCN-DAG: v_bfe_i32 v{{[0-9]+}}, v{{[0-9]+}}, 16, 8

; EG: VTX_READ_32 [[DST:T[0-9]+\.X]], T{{[0-9]+}}.X, 0, #1
; TODO: These should use DST, but for some there are redundant MOVs
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
define amdgpu_kernel void @global_sextload_v3i8_to_v3i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
entry:
  %ld = load <3 x i8>, ptr addrspace(1) %in
  %ext = sext <3 x i8> %ld to <3 x i32>
  store <3 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v4i8_to_v4i32:
; GCN-NOHSA: buffer_load_dword
; GCN-HSA: flat_load_dword

; EG: VTX_READ_32 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0, #1
; TODO: These should use DST, but for some there are redundant MOVs
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
define amdgpu_kernel void @global_zextload_v4i8_to_v4i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <4 x i8>, ptr addrspace(1) %in
  %ext = zext <4 x i8> %load to <4 x i32>
  store <4 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v4i8_to_v4i32:
; GCN-NOHSA: buffer_load_dword
; GCN-HSA: flat_load_dword

; EG: VTX_READ_32 [[DST:T[0-9]+\.X]], T{{[0-9]+}}.X, 0, #1
; TODO: These should use DST, but for some there are redundant MOVs
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
define amdgpu_kernel void @global_sextload_v4i8_to_v4i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <4 x i8>, ptr addrspace(1) %in
  %ext = sext <4 x i8> %load to <4 x i32>
  store <4 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v8i8_to_v8i32:

; EG: VTX_READ_64 T{{[0-9]+}}.XY, T{{[0-9]+}}.X, 0, #1
; TODO: These should use DST, but for some there are redundant MOVs
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
define amdgpu_kernel void @global_zextload_v8i8_to_v8i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <8 x i8>, ptr addrspace(1) %in
  %ext = zext <8 x i8> %load to <8 x i32>
  store <8 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v8i8_to_v8i32:

; EG: VTX_READ_64 [[DST:T[0-9]+\.XY]], T{{[0-9]+}}.X, 0, #1
; TODO: These should use DST, but for some there are redundant MOVs
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
define amdgpu_kernel void @global_sextload_v8i8_to_v8i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <8 x i8>, ptr addrspace(1) %in
  %ext = sext <8 x i8> %load to <8 x i32>
  store <8 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v16i8_to_v16i32:

; EG: VTX_READ_128 T{{[0-9]+}}.XYZW, T{{[0-9]+}}.X, 0, #1
; TODO: These should use DST, but for some there are redundant MOVs
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
define amdgpu_kernel void @global_zextload_v16i8_to_v16i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <16 x i8>, ptr addrspace(1) %in
  %ext = zext <16 x i8> %load to <16 x i32>
  store <16 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v16i8_to_v16i32:

; EG: VTX_READ_128 [[DST:T[0-9]+\.XYZW]], T{{[0-9]+}}.X, 0, #1
; TODO: These should use DST, but for some there are redundant MOVs
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]*.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]*.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]*.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]*.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]*.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]*.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]*.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]*.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]*.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]*.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]*.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]*.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]*.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]*.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]*.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]*.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
define amdgpu_kernel void @global_sextload_v16i8_to_v16i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <16 x i8>, ptr addrspace(1) %in
  %ext = sext <16 x i8> %load to <16 x i32>
  store <16 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v32i8_to_v32i32:

; EG-DAG: VTX_READ_128 T{{[0-9]+}}.XYZW, T{{[0-9]+}}.X, 0, #1
; EG-DAG: VTX_READ_128 T{{[0-9]+}}.XYZW, T{{[0-9]+}}.X, 16, #1
; TODO: These should use DST, but for some there are redundant MOVs
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: BFE_UINT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, {{.*}}literal
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
define amdgpu_kernel void @global_zextload_v32i8_to_v32i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <32 x i8>, ptr addrspace(1) %in
  %ext = zext <32 x i8> %load to <32 x i32>
  store <32 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v32i8_to_v32i32:

; EG-DAG: VTX_READ_128 [[DST_LO:T[0-9]+\.XYZW]], T{{[0-9]+}}.X, 0, #1
; EG-DAG: VTX_READ_128 [[DST_HI:T[0-9]+\.XYZW]], T{{[0-9]+}}.X, 16, #1
; TODO: These should use DST, but for some there are redundant MOVs
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9]+.[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
; EG-DAG: 8
define amdgpu_kernel void @global_sextload_v32i8_to_v32i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <32 x i8>, ptr addrspace(1) %in
  %ext = sext <32 x i8> %load to <32 x i32>
  store <32 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v64i8_to_v64i32:

; EG-DAG: VTX_READ_128 {{T[0-9]+\.XYZW}}, T{{[0-9]+}}.X, 0, #1
; EG-DAG: VTX_READ_128 {{T[0-9]+\.XYZW}}, T{{[0-9]+}}.X, 16, #1
; EG-DAG: VTX_READ_128 {{T[0-9]+\.XYZW}}, T{{[0-9]+}}.X, 32, #1
; EG-DAG: VTX_READ_128 {{T[0-9]+\.XYZW}}, T{{[0-9]+}}.X, 48, #1
define amdgpu_kernel void @global_zextload_v64i8_to_v64i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <64 x i8>, ptr addrspace(1) %in
  %ext = zext <64 x i8> %load to <64 x i32>
  store <64 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v64i8_to_v64i32:

; EG-DAG: VTX_READ_128 {{T[0-9]+\.XYZW}}, T{{[0-9]+}}.X, 0, #1
; EG-DAG: VTX_READ_128 {{T[0-9]+\.XYZW}}, T{{[0-9]+}}.X, 16, #1
; EG-DAG: VTX_READ_128 {{T[0-9]+\.XYZW}}, T{{[0-9]+}}.X, 32, #1
; EG-DAG: VTX_READ_128 {{T[0-9]+\.XYZW}}, T{{[0-9]+}}.X, 48, #1
define amdgpu_kernel void @global_sextload_v64i8_to_v64i32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <64 x i8>, ptr addrspace(1) %in
  %ext = sext <64 x i8> %load to <64 x i32>
  store <64 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_i8_to_i64:
; GCN-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], 0{{$}}

; GCN-NOHSA-DAG: buffer_load_ubyte v[[LO:[0-9]+]],
; GCN-NOHSA: buffer_store_dwordx2 v[[[LO]]:[[HI]]]

; GCN-HSA-DAG: flat_load_ubyte v[[LO:[0-9]+]],
; GCN-HSA: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v[[[LO]]:[[HI]]]

; EG: VTX_READ_8 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0, #1
; EG: MOV {{.*}}, 0.0
define amdgpu_kernel void @global_zextload_i8_to_i64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %a = load i8, ptr addrspace(1) %in
  %ext = zext i8 %a to i64
  store i64 %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_i8_to_i64:
; GCN-NOHSA: buffer_load_sbyte v[[LO:[0-9]+]],
; GCN-HSA: flat_load_sbyte v[[LO:[0-9]+]],
; GCN: v_ashrrev_i32_e32 v[[HI:[0-9]+]], 31, v[[LO]]

; GCN-NOHSA: buffer_store_dwordx2 v[[[LO]]:[[HI]]]
; GCN-HSA: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v[[[LO]]:[[HI]]]

; EG: VTX_READ_8 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0, #1
; EG: ASHR {{\**}} {{T[0-9]\.[XYZW]}}, {{.*}}, literal
; TODO: Why not 7 ?
; EG: 31
define amdgpu_kernel void @global_sextload_i8_to_i64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %a = load i8, ptr addrspace(1) %in
  %ext = sext i8 %a to i64
  store i64 %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v1i8_to_v1i64:

; EG: VTX_READ_8 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0, #1
; EG: MOV {{.*}}, 0.0
define amdgpu_kernel void @global_zextload_v1i8_to_v1i64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <1 x i8>, ptr addrspace(1) %in
  %ext = zext <1 x i8> %load to <1 x i64>
  store <1 x i64> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v1i8_to_v1i64:

; EG: VTX_READ_8 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0, #1
; EG: ASHR {{\**}} {{T[0-9]\.[XYZW]}}, {{.*}}, literal
; TODO: Why not 7 ?
; EG: 31
define amdgpu_kernel void @global_sextload_v1i8_to_v1i64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <1 x i8>, ptr addrspace(1) %in
  %ext = sext <1 x i8> %load to <1 x i64>
  store <1 x i64> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v2i8_to_v2i64:

; EG: VTX_READ_16 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0, #1
define amdgpu_kernel void @global_zextload_v2i8_to_v2i64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <2 x i8>, ptr addrspace(1) %in
  %ext = zext <2 x i8> %load to <2 x i64>
  store <2 x i64> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v2i8_to_v2i64:

; EG: VTX_READ_16 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0, #1
define amdgpu_kernel void @global_sextload_v2i8_to_v2i64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <2 x i8>, ptr addrspace(1) %in
  %ext = sext <2 x i8> %load to <2 x i64>
  store <2 x i64> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v4i8_to_v4i64:

; EG: VTX_READ_32 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0, #1
define amdgpu_kernel void @global_zextload_v4i8_to_v4i64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <4 x i8>, ptr addrspace(1) %in
  %ext = zext <4 x i8> %load to <4 x i64>
  store <4 x i64> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v4i8_to_v4i64:

; EG: VTX_READ_32 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0, #1
define amdgpu_kernel void @global_sextload_v4i8_to_v4i64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <4 x i8>, ptr addrspace(1) %in
  %ext = sext <4 x i8> %load to <4 x i64>
  store <4 x i64> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v8i8_to_v8i64:

; EG: VTX_READ_64 T{{[0-9]+}}.XY, T{{[0-9]+}}.X, 0, #1
define amdgpu_kernel void @global_zextload_v8i8_to_v8i64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <8 x i8>, ptr addrspace(1) %in
  %ext = zext <8 x i8> %load to <8 x i64>
  store <8 x i64> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v8i8_to_v8i64:

; EG: VTX_READ_64 T{{[0-9]+}}.XY, T{{[0-9]+}}.X, 0, #1
define amdgpu_kernel void @global_sextload_v8i8_to_v8i64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <8 x i8>, ptr addrspace(1) %in
  %ext = sext <8 x i8> %load to <8 x i64>
  store <8 x i64> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v16i8_to_v16i64:

; EG: VTX_READ_128 T{{[0-9]+}}.XYZW, T{{[0-9]+}}.X, 0, #1
define amdgpu_kernel void @global_zextload_v16i8_to_v16i64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <16 x i8>, ptr addrspace(1) %in
  %ext = zext <16 x i8> %load to <16 x i64>
  store <16 x i64> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v16i8_to_v16i64:

; EG: VTX_READ_128 T{{[0-9]+}}.XYZW, T{{[0-9]+}}.X, 0, #1
define amdgpu_kernel void @global_sextload_v16i8_to_v16i64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <16 x i8>, ptr addrspace(1) %in
  %ext = sext <16 x i8> %load to <16 x i64>
  store <16 x i64> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v32i8_to_v32i64:

; EG-DAG: VTX_READ_128 T{{[0-9]+}}.XYZW, T{{[0-9]+}}.X, 0, #1
; EG-DAG: VTX_READ_128 T{{[0-9]+}}.XYZW, T{{[0-9]+}}.X, 16, #1
define amdgpu_kernel void @global_zextload_v32i8_to_v32i64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <32 x i8>, ptr addrspace(1) %in
  %ext = zext <32 x i8> %load to <32 x i64>
  store <32 x i64> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v32i8_to_v32i64:

; EG-DAG: VTX_READ_128 T{{[0-9]+}}.XYZW, T{{[0-9]+}}.X, 0, #1
; EG-DAG: VTX_READ_128 T{{[0-9]+}}.XYZW, T{{[0-9]+}}.X, 16, #1
define amdgpu_kernel void @global_sextload_v32i8_to_v32i64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <32 x i8>, ptr addrspace(1) %in
  %ext = sext <32 x i8> %load to <32 x i64>
  store <32 x i64> %ext, ptr addrspace(1) %out
  ret void
}

; XFUNC-LABEL: {{^}}global_zextload_v64i8_to_v64i64:
; define amdgpu_kernel void @global_zextload_v64i8_to_v64i64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
;   %load = load <64 x i8>, ptr addrspace(1) %in
;   %ext = zext <64 x i8> %load to <64 x i64>
;   store <64 x i64> %ext, ptr addrspace(1) %out
;   ret void
; }

; XFUNC-LABEL: {{^}}global_sextload_v64i8_to_v64i64:
; define amdgpu_kernel void @global_sextload_v64i8_to_v64i64(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
;   %load = load <64 x i8>, ptr addrspace(1) %in
;   %ext = sext <64 x i8> %load to <64 x i64>
;   store <64 x i64> %ext, ptr addrspace(1) %out
;   ret void
; }

; FUNC-LABEL: {{^}}global_zextload_i8_to_i16:
; GCN-NOHSA: buffer_load_ubyte v[[VAL:[0-9]+]],
; GCN-NOHSA: buffer_store_short v[[VAL]]

; GCN-HSA: flat_load_ubyte v[[VAL:[0-9]+]],
; GCN-HSA: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, v[[VAL]]

; EG: VTX_READ_8 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0, #1
define amdgpu_kernel void @global_zextload_i8_to_i16(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %a = load i8, ptr addrspace(1) %in
  %ext = zext i8 %a to i16
  store i16 %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_i8_to_i16:
; GCN-NOHSA: buffer_load_sbyte v[[VAL:[0-9]+]],
; GCN-HSA: flat_load_sbyte v[[VAL:[0-9]+]],

; GCN-NOHSA: buffer_store_short v[[VAL]]
; GCN-HSA: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, v[[VAL]]

; EG: VTX_READ_8 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0, #1
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
define amdgpu_kernel void @global_sextload_i8_to_i16(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %a = load i8, ptr addrspace(1) %in
  %ext = sext i8 %a to i16
  store i16 %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v1i8_to_v1i16:

; EG: VTX_READ_8 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0, #1
define amdgpu_kernel void @global_zextload_v1i8_to_v1i16(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <1 x i8>, ptr addrspace(1) %in
  %ext = zext <1 x i8> %load to <1 x i16>
  store <1 x i16> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v1i8_to_v1i16:

; EG: VTX_READ_8 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0, #1
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
define amdgpu_kernel void @global_sextload_v1i8_to_v1i16(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <1 x i8>, ptr addrspace(1) %in
  %ext = sext <1 x i8> %load to <1 x i16>
  store <1 x i16> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v2i8_to_v2i16:

; EG: VTX_READ_16 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0, #1
define amdgpu_kernel void @global_zextload_v2i8_to_v2i16(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <2 x i8>, ptr addrspace(1) %in
  %ext = zext <2 x i8> %load to <2 x i16>
  store <2 x i16> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v2i8_to_v2i16:

; EG: VTX_READ_16 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0, #1
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
define amdgpu_kernel void @global_sextload_v2i8_to_v2i16(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <2 x i8>, ptr addrspace(1) %in
  %ext = sext <2 x i8> %load to <2 x i16>
  store <2 x i16> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v4i8_to_v4i16:

; EG: VTX_READ_32 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0, #1
define amdgpu_kernel void @global_zextload_v4i8_to_v4i16(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <4 x i8>, ptr addrspace(1) %in
  %ext = zext <4 x i8> %load to <4 x i16>
  store <4 x i16> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v4i8_to_v4i16:

; EG: VTX_READ_32 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0, #1
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
define amdgpu_kernel void @global_sextload_v4i8_to_v4i16(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <4 x i8>, ptr addrspace(1) %in
  %ext = sext <4 x i8> %load to <4 x i16>
  store <4 x i16> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v8i8_to_v8i16:

; EG: VTX_READ_64 T{{[0-9]+}}.XY, T{{[0-9]+}}.X, 0, #1
define amdgpu_kernel void @global_zextload_v8i8_to_v8i16(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <8 x i8>, ptr addrspace(1) %in
  %ext = zext <8 x i8> %load to <8 x i16>
  store <8 x i16> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v8i8_to_v8i16:

; EG: VTX_READ_64 T{{[0-9]+}}.XY, T{{[0-9]+}}.X, 0, #1
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
define amdgpu_kernel void @global_sextload_v8i8_to_v8i16(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <8 x i8>, ptr addrspace(1) %in
  %ext = sext <8 x i8> %load to <8 x i16>
  store <8 x i16> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v16i8_to_v16i16:

; EG: VTX_READ_128 T{{[0-9]+}}.XYZW, T{{[0-9]+}}.X, 0, #1
define amdgpu_kernel void @global_zextload_v16i8_to_v16i16(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <16 x i8>, ptr addrspace(1) %in
  %ext = zext <16 x i8> %load to <16 x i16>
  store <16 x i16> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v16i8_to_v16i16:

; EG: VTX_READ_128 T{{[0-9]+}}.XYZW, T{{[0-9]+}}.X, 0, #1
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
define amdgpu_kernel void @global_sextload_v16i8_to_v16i16(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <16 x i8>, ptr addrspace(1) %in
  %ext = sext <16 x i8> %load to <16 x i16>
  store <16 x i16> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_zextload_v32i8_to_v32i16:

; EG-DAG: VTX_READ_128 T{{[0-9]+}}.XYZW, T{{[0-9]+}}.X, 0, #1
; EG-DAG: VTX_READ_128 T{{[0-9]+}}.XYZW, T{{[0-9]+}}.X, 16, #1
define amdgpu_kernel void @global_zextload_v32i8_to_v32i16(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <32 x i8>, ptr addrspace(1) %in
  %ext = zext <32 x i8> %load to <32 x i16>
  store <32 x i16> %ext, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_sextload_v32i8_to_v32i16:

; EG-DAG: VTX_READ_128 T{{[0-9]+}}.XYZW, T{{[0-9]+}}.X, 0, #1
; EG-DAG: VTX_READ_128 T{{[0-9]+}}.XYZW, T{{[0-9]+}}.X, 16, #1
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
; EG-DAG: BFE_INT {{[* ]*}}T{{[0-9].[XYZW]}}, {{.*}}, 0.0, literal
define amdgpu_kernel void @global_sextload_v32i8_to_v32i16(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %load = load <32 x i8>, ptr addrspace(1) %in
  %ext = sext <32 x i8> %load to <32 x i16>
  store <32 x i16> %ext, ptr addrspace(1) %out
  ret void
}

; XFUNC-LABEL: {{^}}global_zextload_v64i8_to_v64i16:
; define amdgpu_kernel void @global_zextload_v64i8_to_v64i16(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
;   %load = load <64 x i8>, ptr addrspace(1) %in
;   %ext = zext <64 x i8> %load to <64 x i16>
;   store <64 x i16> %ext, ptr addrspace(1) %out
;   ret void
; }

; XFUNC-LABEL: {{^}}global_sextload_v64i8_to_v64i16:
; define amdgpu_kernel void @global_sextload_v64i8_to_v64i16(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
;   %load = load <64 x i8>, ptr addrspace(1) %in
;   %ext = sext <64 x i8> %load to <64 x i16>
;   store <64 x i16> %ext, ptr addrspace(1) %out
;   ret void
; }

attributes #0 = { nounwind }
