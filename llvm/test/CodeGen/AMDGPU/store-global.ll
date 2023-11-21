; RUN: llc -mtriple=amdgcn-- -mcpu=verde -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SIVI -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -mtriple=amdgcn-- -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SIVI -check-prefix=VI -check-prefix=FUNC %s
; RUN: llc -mtriple=amdgcn-- -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GFX9 -check-prefix=FUNC %s
; RUN: llc -march=r600 -mtriple=r600-- -mcpu=redwood -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s
; RUN: llc -march=r600 -mtriple=r600-- -mcpu=cayman -verify-machineinstrs < %s | FileCheck -check-prefix=CM -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}store_i1:
; EG: MEM_RAT MSKOR
; EG-NOT: MEM_RAT MSKOR

; CM: MEM_RAT MSKOR
; CM-NOT: MEM_RAT MSKOR

; SIVI: buffer_store_byte
; GFX9: global_store_byte
define amdgpu_kernel void @store_i1(ptr addrspace(1) %out) {
entry:
  store i1 true, ptr addrspace(1) %out
  ret void
}

; i8 store
; FUNC-LABEL: {{^}}store_i8:
; EG: MEM_RAT MSKOR T[[RW_GPR:[0-9]]].XW, T{{[0-9]}}.X
; EG-NOT: MEM_RAT MSKOR

; EG: VTX_READ_8
; EG: AND_INT
; EG: AND_INT
; EG: LSHL
; EG: LSHL
; EG: LSHL

; SIVI: buffer_store_byte
; GFX9: global_store_byte
define amdgpu_kernel void @store_i8(ptr addrspace(1) %out, i8 %in) {
entry:
  store i8 %in, ptr addrspace(1) %out
  ret void
}

; i16 store
; FUNC-LABEL: {{^}}store_i16:
; EG: MEM_RAT MSKOR T[[RW_GPR:[0-9]]].XW, T{{[0-9]}}.X
; EG-NOT: MEM_RAT MSKOR

; EG: VTX_READ_16
; EG: AND_INT
; EG: AND_INT
; EG: LSHL
; EG: LSHL
; EG: LSHL


; SIVI: buffer_store_short
; GFX9: global_store_short
define amdgpu_kernel void @store_i16(ptr addrspace(1) %out, i16 %in) {
entry:
  store i16 %in, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}store_i24:
; SIVI: s_lshr_b32 s{{[0-9]+}}, s{{[0-9]+}}, 16
; SIVI-DAG: buffer_store_byte
; SIVI-DAG: buffer_store_short

; GFX9-DAG: global_store_byte_d16_hi v{{[0-9]+}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}} offset:2
; GFX9-DAG: global_store_short

; EG: MEM_RAT MSKOR
; EG: MEM_RAT MSKOR
define amdgpu_kernel void @store_i24(ptr addrspace(1) %out, i24 %in) {
entry:
  store i24 %in, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}store_i25:
; GCN: s_and_b32 [[AND:s[0-9]+]], s{{[0-9]+}}, 0x1ffffff{{$}}
; GCN: v_mov_b32_e32 [[VAND:v[0-9]+]], [[AND]]
; SIVI: buffer_store_dword [[VAND]]
; GFX9: global_store_dword v{{[0-9]+}}, [[VAND]], s

; EG: MEM_RAT_CACHELESS STORE_RAW
; EG-NOT: MEM_RAT

; CM: MEM_RAT_CACHELESS STORE_DWORD
; CM-NOT: MEM_RAT
define amdgpu_kernel void @store_i25(ptr addrspace(1) %out, i25 %in) {
entry:
  store i25 %in, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}store_v2i8:
; v2i8 is naturally 2B aligned
; EG: MEM_RAT MSKOR
; EG-NOT: MEM_RAT MSKOR

; CM: MEM_RAT MSKOR
; CM-NOT: MEM_RAT MSKOR

; SIVI: buffer_store_short
; GFX9: global_store_short
define amdgpu_kernel void @store_v2i8(ptr addrspace(1) %out, <2 x i32> %in) {
entry:
  %0 = trunc <2 x i32> %in to <2 x i8>
  store <2 x i8> %0, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}store_v2i8_unaligned:
; EG: MEM_RAT MSKOR
; EG: MEM_RAT MSKOR
; EG-NOT: MEM_RAT MSKOR

; CM: MEM_RAT MSKOR
; CM: MEM_RAT MSKOR
; CM-NOT: MEM_RAT MSKOR

; SI: buffer_store_byte
define amdgpu_kernel void @store_v2i8_unaligned(ptr addrspace(1) %out, <2 x i32> %in) {
entry:
  %0 = trunc <2 x i32> %in to <2 x i8>
  store <2 x i8> %0, ptr addrspace(1) %out, align 1
  ret void
}


; FUNC-LABEL: {{^}}store_v2i16:
; EG: MEM_RAT_CACHELESS STORE_RAW

; CM: MEM_RAT_CACHELESS STORE_DWORD

; SIVI: buffer_store_dword
; GFX9: global_store_dword
define amdgpu_kernel void @store_v2i16(ptr addrspace(1) %out, <2 x i32> %in) {
entry:
  %0 = trunc <2 x i32> %in to <2 x i16>
  store <2 x i16> %0, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}store_v2i16_unaligned:
; EG: MEM_RAT MSKOR
; EG: MEM_RAT MSKOR
; EG-NOT: MEM_RAT MSKOR
; EG-NOT: MEM_RAT_CACHELESS STORE_RAW

; CM: MEM_RAT MSKOR
; CM: MEM_RAT MSKOR
; CM-NOT: MEM_RAT MSKOR
; CM-NOT: MEM_RAT_CACHELESS STORE_DWORD

; SIVI: buffer_store_short
; SIVI: buffer_store_short

; GFX9: global_store_short
; GFX9: global_store_short
define amdgpu_kernel void @store_v2i16_unaligned(ptr addrspace(1) %out, <2 x i32> %in) {
entry:
  %0 = trunc <2 x i32> %in to <2 x i16>
  store <2 x i16> %0, ptr addrspace(1) %out, align 2
  ret void
}

; FUNC-LABEL: {{^}}store_v4i8:
; EG: MEM_RAT_CACHELESS STORE_RAW

; CM: MEM_RAT_CACHELESS STORE_DWORD

; SIVI: buffer_store_dword
; GFX9: global_store_dword
define amdgpu_kernel void @store_v4i8(ptr addrspace(1) %out, <4 x i32> %in) {
entry:
  %0 = trunc <4 x i32> %in to <4 x i8>
  store <4 x i8> %0, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}store_v4i8_unaligned:
; EG: MEM_RAT MSKOR
; EG: MEM_RAT MSKOR
; EG: MEM_RAT MSKOR
; EG: MEM_RAT MSKOR
; EG-NOT: MEM_RAT MSKOR
; EG-NOT: MEM_RAT_CACHELESS STORE_RAW

; CM: MEM_RAT MSKOR
; CM: MEM_RAT MSKOR
; CM: MEM_RAT MSKOR
; CM: MEM_RAT MSKOR
; CM-NOT: MEM_RAT MSKOR
; CM-NOT: MEM_RAT_CACHELESS STORE_DWORD

; SI: buffer_store_byte
; SI: buffer_store_byte
; SI: buffer_store_byte
; SI: buffer_store_byte
; SI-NOT: buffer_store_dword
define amdgpu_kernel void @store_v4i8_unaligned(ptr addrspace(1) %out, <4 x i32> %in) {
entry:
  %0 = trunc <4 x i32> %in to <4 x i8>
  store <4 x i8> %0, ptr addrspace(1) %out, align 1
  ret void
}

; FUNC-LABEL: {{^}}store_v4i8_halfaligned:
; EG: MEM_RAT MSKOR
; EG: MEM_RAT MSKOR
; EG-NOT: MEM_RAT MSKOR
; EG-NOT: MEM_RAT_CACHELESS STORE_RAW

; CM: MEM_RAT MSKOR
; CM: MEM_RAT MSKOR
; CM-NOT: MEM_RAT MSKOR
; CM-NOT: MEM_RAT_CACHELESS STORE_DWORD

; SI: buffer_store_short
; SI: buffer_store_short
; SI-NOT: buffer_store_dword
define amdgpu_kernel void @store_v4i8_halfaligned(ptr addrspace(1) %out, <4 x i32> %in) {
entry:
  %0 = trunc <4 x i32> %in to <4 x i8>
  store <4 x i8> %0, ptr addrspace(1) %out, align 2
  ret void
}

; floating-point store
; FUNC-LABEL: {{^}}store_f32:
; EG: MEM_RAT_CACHELESS STORE_RAW T{{[0-9]+\.X, T[0-9]+\.X}}, 1

; CM: MEM_RAT_CACHELESS STORE_DWORD T{{[0-9]+\.X, T[0-9]+\.X}}

; SIVI: buffer_store_dword
; GFX9: global_store_dword

define amdgpu_kernel void @store_f32(ptr addrspace(1) %out, float %in) {
  store float %in, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}store_v4i16:
; EG: MEM_RAT_CACHELESS STORE_RAW T{{[0-9]+}}.XY

; CM: MEM_RAT_CACHELESS STORE_DWORD T{{[0-9]+}}

; SIVI: buffer_store_dwordx2
; GFX9: global_store_dwordx2
define amdgpu_kernel void @store_v4i16(ptr addrspace(1) %out, <4 x i32> %in) {
entry:
  %0 = trunc <4 x i32> %in to <4 x i16>
  store <4 x i16> %0, ptr addrspace(1) %out
  ret void
}

; vec2 floating-point stores
; FUNC-LABEL: {{^}}store_v2f32:
; EG: MEM_RAT_CACHELESS STORE_RAW

; CM: MEM_RAT_CACHELESS STORE_DWORD

; SIVI: buffer_store_dwordx2
; GFX9: global_store_dwordx2

define amdgpu_kernel void @store_v2f32(ptr addrspace(1) %out, float %a, float %b) {
entry:
  %0 = insertelement <2 x float> <float 0.0, float 0.0>, float %a, i32 0
  %1 = insertelement <2 x float> %0, float %b, i32 1
  store <2 x float> %1, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}store_v3i32:
; SI-DAG: buffer_store_dword v
; SI-DAG: buffer_store_dwordx2

; VI: buffer_store_dwordx3

; GFX9: global_store_dwordx3

; EG-DAG: MEM_RAT_CACHELESS STORE_RAW {{T[0-9]+\.[XYZW]}}, {{T[0-9]+\.[XYZW]}},
; EG-DAG: MEM_RAT_CACHELESS STORE_RAW {{T[0-9]+\.XY}}, {{T[0-9]+\.[XYZW]}},
define amdgpu_kernel void @store_v3i32(ptr addrspace(1) %out, <3 x i32> %a) nounwind {
  store <3 x i32> %a, ptr addrspace(1) %out, align 16
  ret void
}

; FUNC-LABEL: {{^}}store_v4i32:
; EG: MEM_RAT_CACHELESS STORE_RAW {{T[0-9]+\.XYZW}}
; EG-NOT: MEM_RAT_CACHELESS STORE_RAW

; CM: MEM_RAT_CACHELESS STORE_DWORD
; CM-NOT: MEM_RAT_CACHELESS STORE_DWORD

; SIVI: buffer_store_dwordx4
; GFX9: global_store_dwordx4
define amdgpu_kernel void @store_v4i32(ptr addrspace(1) %out, <4 x i32> %in) {
entry:
  store <4 x i32> %in, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}store_v4i32_unaligned:
; EG: MEM_RAT_CACHELESS STORE_RAW {{T[0-9]+\.XYZW}}
; EG-NOT: MEM_RAT_CACHELESS STORE_RAW

; CM: MEM_RAT_CACHELESS STORE_DWORD
; CM-NOT: MEM_RAT_CACHELESS STORE_DWORD

; SIVI: buffer_store_dwordx4
; GFX9: global_store_dwordx4
define amdgpu_kernel void @store_v4i32_unaligned(ptr addrspace(1) %out, <4 x i32> %in) {
entry:
  store <4 x i32> %in, ptr addrspace(1) %out, align 4
  ret void
}

; v4f32 store
; FUNC-LABEL: {{^}}store_v4f32:
; EG: MEM_RAT_CACHELESS STORE_RAW T{{[0-9]+\.XYZW, T[0-9]+\.X}}, 1
; EG-NOT: MEM_RAT_CACHELESS STORE_RAW

; CM: MEM_RAT_CACHELESS STORE_DWORD
; CM-NOT: MEM_RAT_CACHELESS STORE_DWORD

; SIVI: buffer_store_dwordx4
; GFX9: global_store_dwordx4
define amdgpu_kernel void @store_v4f32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %1 = load <4 x float>, ptr addrspace(1) %in
  store <4 x float> %1, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}store_i64_i8:
; EG: MEM_RAT MSKOR

; CM: MEM_RAT MSKOR

; SIVI: buffer_store_byte
; GFX9: global_store_byte
define amdgpu_kernel void @store_i64_i8(ptr addrspace(1) %out, i64 %in) {
entry:
  %0 = trunc i64 %in to i8
  store i8 %0, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}store_i64_i16:
; EG: MEM_RAT MSKOR
; SIVI: buffer_store_short
; GFX9: global_store_short
define amdgpu_kernel void @store_i64_i16(ptr addrspace(1) %out, i64 %in) {
entry:
  %0 = trunc i64 %in to i16
  store i16 %0, ptr addrspace(1) %out
  ret void
}

; The stores in this function are combined by the optimizer to create a
; 64-bit store with 32-bit alignment.  This is legal and the legalizer
; should not try to split the 64-bit store back into 2 32-bit stores.

; FUNC-LABEL: {{^}}vecload2:
; EG: MEM_RAT_CACHELESS STORE_RAW T{{[0-9]+\.XY, T[0-9]+\.X}}, 1
; EG-NOT: MEM_RAT_CACHELESS STORE_RAW

; CM: MEM_RAT_CACHELESS STORE_DWORD
; CM-NOT: MEM_RAT_CACHELESS STORE_DWORD

; SIVI: buffer_store_dwordx2
; GFX9: global_store_dwordx2
define amdgpu_kernel void @vecload2(ptr addrspace(1) nocapture %out, ptr addrspace(4) nocapture %mem) #0 {
entry:
  %0 = load i32, ptr addrspace(4) %mem, align 4
  %arrayidx1.i = getelementptr inbounds i32, ptr addrspace(4) %mem, i64 1
  %1 = load i32, ptr addrspace(4) %arrayidx1.i, align 4
  store i32 %0, ptr addrspace(1) %out, align 4
  %arrayidx1 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 1
  store i32 %1, ptr addrspace(1) %arrayidx1, align 4
  ret void
}

; When i128 was a legal type this program generated cannot select errors:

; FUNC-LABEL: {{^}}"i128-const-store":
; EG: MEM_RAT_CACHELESS STORE_RAW T{{[0-9]+}}.XYZW, T{{[0-9]+}}.X, 1

; CM: MEM_RAT_CACHELESS STORE_DWORD T{{[0-9]+}}, T{{[0-9]+}}.X

; SIVI: buffer_store_dwordx4
; GFX9: global_store_dwordx4
define amdgpu_kernel void @i128-const-store(ptr addrspace(1) %out) {
entry:
  store i32 1, ptr addrspace(1) %out, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 1
  store i32 1, ptr addrspace(1) %arrayidx2, align 4
  %arrayidx4 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 2
  store i32 2, ptr addrspace(1) %arrayidx4, align 4
  %arrayidx6 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 3
  store i32 2, ptr addrspace(1) %arrayidx6, align 4
  ret void
}

attributes #0 = { nounwind }
