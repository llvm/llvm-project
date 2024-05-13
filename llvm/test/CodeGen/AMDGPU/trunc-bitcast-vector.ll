; RUN: llc -amdgpu-scalarize-global-loads=false -mtriple=amdgcn -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,SI %s
; RUN: llc -amdgpu-scalarize-global-loads=false -mtriple=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,VI %s

; GCN-LABEL: {{^}}trunc_i64_bitcast_v2i32:
; GCN: buffer_load_dword v
; GCN: buffer_store_dword v
define amdgpu_kernel void @trunc_i64_bitcast_v2i32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %ld = load <2 x i32>, ptr addrspace(1) %in
  %bc = bitcast <2 x i32> %ld to i64
  %trunc = trunc i64 %bc to i32
  store i32 %trunc, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}trunc_i96_bitcast_v3i32:
; GCN: buffer_load_dword v
; GCN: buffer_store_dword v
define amdgpu_kernel void @trunc_i96_bitcast_v3i32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %ld = load <3 x i32>, ptr addrspace(1) %in
  %bc = bitcast <3 x i32> %ld to i96
  %trunc = trunc i96 %bc to i32
  store i32 %trunc, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}trunc_i128_bitcast_v4i32:
; GCN: buffer_load_dword v
; GCN: buffer_store_dword v
define amdgpu_kernel void @trunc_i128_bitcast_v4i32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %ld = load <4 x i32>, ptr addrspace(1) %in
  %bc = bitcast <4 x i32> %ld to i128
  %trunc = trunc i128 %bc to i32
  store i32 %trunc, ptr addrspace(1) %out
  ret void
}

; Don't want load width reduced in this case.
; GCN-LABEL: {{^}}trunc_i16_bitcast_v2i16:
; GCN: buffer_load_dword [[VAL:v[0-9]+]]
; GCN: buffer_store_short [[VAL]]
define amdgpu_kernel void @trunc_i16_bitcast_v2i16(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %ld = load <2 x i16>, ptr addrspace(1) %in
  %bc = bitcast <2 x i16> %ld to i32
  %trunc = trunc i32 %bc to i16
  store i16 %trunc, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}trunc_i16_bitcast_v4i16:
; FIXME We need to teach the dagcombiner to reduce load width for:
;   t21: v2i32,ch = load<LD8[%in(addrspace=1)]> t12, t10, undef:i64
;        t23: i64 = bitcast t21
;      t30: i16 = truncate t23
; GCN: buffer_load_dword v[[VAL:[0-9]+]]
; GCN: buffer_store_short v[[VAL]], off
define amdgpu_kernel void @trunc_i16_bitcast_v4i16(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %ld = load <4 x i16>, ptr addrspace(1) %in
  %bc = bitcast <4 x i16> %ld to i64
  %trunc = trunc i64 %bc to i16
  store i16 %trunc, ptr addrspace(1) %out
  ret void
}

; FIXME: Consistently shrink or not here
; GCN-LABEL: {{^}}trunc_i8_bitcast_v2i8:
; SI: buffer_load_ubyte [[VAL:v[0-9]+]]
; VI: buffer_load_ushort [[VAL:v[0-9]+]]
; GCN: buffer_store_byte [[VAL]]
define amdgpu_kernel void @trunc_i8_bitcast_v2i8(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %ld = load <2 x i8>, ptr addrspace(1) %in
  %bc = bitcast <2 x i8> %ld to i16
  %trunc = trunc i16 %bc to i8
  store i8 %trunc, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}trunc_i32_bitcast_v4i8:
; GCN: buffer_load_dword [[VAL:v[0-9]+]]
; GCN: buffer_store_byte [[VAL]]
define amdgpu_kernel void @trunc_i32_bitcast_v4i8(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %ld = load <4 x i8>, ptr addrspace(1) %in
  %bc = bitcast <4 x i8> %ld to i32
  %trunc = trunc i32 %bc to i8
  store i8 %trunc, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}trunc_i24_bitcast_v3i8:
; GCN: buffer_load_dword [[VAL:v[0-9]+]]
; GCN: buffer_store_byte [[VAL]]
define amdgpu_kernel void @trunc_i24_bitcast_v3i8(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %ld = load <3 x i8>, ptr addrspace(1) %in
  %bc = bitcast <3 x i8> %ld to i24
  %trunc = trunc i24 %bc to i8
  store i8 %trunc, ptr addrspace(1) %out
  ret void
}
