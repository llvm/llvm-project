; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -verify-machineinstrs < %s | FileCheck --check-prefix=SI %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck --check-prefix=VI %s

; CHECK-LABEL: {{^}}trunc_i64_bitcast_v2i32:
; CHECK: buffer_load_dword v
; CHECK: buffer_store_dword v
define amdgpu_kernel void @trunc_i64_bitcast_v2i32(i32 addrspace(1)* %out, <2 x i32> addrspace(1)* %in) {
  %ld = load <2 x i32>, <2 x i32> addrspace(1)* %in
  %bc = bitcast <2 x i32> %ld to i64
  %trunc = trunc i64 %bc to i32
  store i32 %trunc, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}trunc_i96_bitcast_v3i32:
; CHECK: buffer_load_dword v
; CHECK: buffer_store_dword v
define amdgpu_kernel void @trunc_i96_bitcast_v3i32(i32 addrspace(1)* %out, <3 x i32> addrspace(1)* %in) {
  %ld = load <3 x i32>, <3 x i32> addrspace(1)* %in
  %bc = bitcast <3 x i32> %ld to i96
  %trunc = trunc i96 %bc to i32
  store i32 %trunc, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}trunc_i128_bitcast_v4i32:
; CHECK: buffer_load_dword v
; CHECK: buffer_store_dword v
define amdgpu_kernel void @trunc_i128_bitcast_v4i32(i32 addrspace(1)* %out, <4 x i32> addrspace(1)* %in) {
  %ld = load <4 x i32>, <4 x i32> addrspace(1)* %in
  %bc = bitcast <4 x i32> %ld to i128
  %trunc = trunc i128 %bc to i32
  store i32 %trunc, i32 addrspace(1)* %out
  ret void
}

; Don't want load width reduced in this case.
; CHECK-LABEL: {{^}}trunc_i16_bitcast_v2i16:
; CHECK: buffer_load_dword [[VAL:v[0-9]+]]
; CHECK: buffer_store_short [[VAL]]
define amdgpu_kernel void @trunc_i16_bitcast_v2i16(i16 addrspace(1)* %out, <2 x i16> addrspace(1)* %in) {
  %ld = load <2 x i16>, <2 x i16> addrspace(1)* %in
  %bc = bitcast <2 x i16> %ld to i32
  %trunc = trunc i32 %bc to i16
  store i16 %trunc, i16 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}trunc_i16_bitcast_v4i16:
; FIXME We need to teach the dagcombiner to reduce load width for:
;   t21: v2i32,ch = load<LD8[%in(addrspace=1)]> t12, t10, undef:i64
;        t23: i64 = bitcast t21
;      t30: i16 = truncate t23
; SI: buffer_load_dword v[[VAL:[0-9]+]]
; VI: buffer_load_dwordx2 v{{\[}}[[VAL:[0-9]+]]
; CHECK: buffer_store_short [[VAL]]
define amdgpu_kernel void @trunc_i16_bitcast_v4i16(i16 addrspace(1)* %out, <4 x i16> addrspace(1)* %in) {
  %ld = load <4 x i16>, <4 x i16> addrspace(1)* %in
  %bc = bitcast <4 x i16> %ld to i64
  %trunc = trunc i64 %bc to i16
  store i16 %trunc, i16 addrspace(1)* %out
  ret void
}

; FIXME: Don't want load width reduced in this case.
; CHECK-LABEL: {{^}}trunc_i8_bitcast_v2i8:
; CHECK: buffer_load_ubyte [[VAL:v[0-9]+]]
; CHECK: buffer_store_byte [[VAL]]
define amdgpu_kernel void @trunc_i8_bitcast_v2i8(i8 addrspace(1)* %out, <2 x i8> addrspace(1)* %in) {
  %ld = load <2 x i8>, <2 x i8> addrspace(1)* %in
  %bc = bitcast <2 x i8> %ld to i16
  %trunc = trunc i16 %bc to i8
  store i8 %trunc, i8 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}trunc_i32_bitcast_v4i8:
; CHECK: buffer_load_dword [[VAL:v[0-9]+]]
; CHECK: buffer_store_byte [[VAL]]
define amdgpu_kernel void @trunc_i32_bitcast_v4i8(i8 addrspace(1)* %out, <4 x i8> addrspace(1)* %in) {
  %ld = load <4 x i8>, <4 x i8> addrspace(1)* %in
  %bc = bitcast <4 x i8> %ld to i32
  %trunc = trunc i32 %bc to i8
  store i8 %trunc, i8 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: {{^}}trunc_i24_bitcast_v3i8:
; CHECK: buffer_load_dword [[VAL:v[0-9]+]]
; CHECK: buffer_store_byte [[VAL]]
define amdgpu_kernel void @trunc_i24_bitcast_v3i8(i8 addrspace(1)* %out, <3 x i8> addrspace(1)* %in) {
  %ld = load <3 x i8>, <3 x i8> addrspace(1)* %in
  %bc = bitcast <3 x i8> %ld to i24
  %trunc = trunc i24 %bc to i8
  store i8 %trunc, i8 addrspace(1)* %out
  ret void
}
