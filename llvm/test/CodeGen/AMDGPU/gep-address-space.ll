; RUN: llc -march=amdgcn -verify-machineinstrs< %s | FileCheck --check-prefix=SI --check-prefix=CHECK %s
; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs< %s | FileCheck --check-prefix=CI --check-prefix=CHECK %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs< %s | FileCheck --check-prefix=CI --check-prefix=CHECK %s

define amdgpu_kernel void @use_gep_address_space([1024 x i32] addrspace(3)* %array) nounwind {
; CHECK-LABEL: {{^}}use_gep_address_space:
; CHECK: v_mov_b32_e32 [[PTR:v[0-9]+]], s{{[0-9]+}}
; CHECK: ds_write_b32 [[PTR]], v{{[0-9]+}} offset:64
  %p = getelementptr [1024 x i32], [1024 x i32] addrspace(3)* %array, i16 0, i16 16
  store i32 99, i32 addrspace(3)* %p
  ret void
}

; CHECK-LABEL: {{^}}use_gep_address_space_large_offset:
; The LDS offset will be 65536 bytes, which is larger than the size of LDS on
; SI, which is why it is being OR'd with the base pointer.
; SI: s_or_b32
; CI: s_add_i32
; CHECK: ds_write_b32
define amdgpu_kernel void @use_gep_address_space_large_offset([1024 x i32] addrspace(3)* %array) nounwind {
  %p = getelementptr [1024 x i32], [1024 x i32] addrspace(3)* %array, i16 0, i16 16384
  store i32 99, i32 addrspace(3)* %p
  ret void
}

; CHECK-LABEL: {{^}}gep_as_vector_v4:
; SI: s_add_i32
; SI: s_add_i32
; SI: s_add_i32
; SI: s_add_i32

; CHECK-DAG: v_mov_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}
; CHECK-DAG: v_mov_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}
; CHECK-DAG: v_mov_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}
; CHECK-DAG: v_mov_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}

; CI-DAG: ds_write_b32 v{{[0-9]+}}, v{{[0-9]+}} offset:64
; CI-DAG: ds_write_b32 v{{[0-9]+}}, v{{[0-9]+}} offset:64
; CI-DAG: ds_write_b32 v{{[0-9]+}}, v{{[0-9]+}} offset:64
; CI-DAG: ds_write_b32 v{{[0-9]+}}, v{{[0-9]+}} offset:64
; CHECK: s_endpgm
define amdgpu_kernel void @gep_as_vector_v4(<4 x [1024 x i32] addrspace(3)*> %array) nounwind {
  %p = getelementptr [1024 x i32], <4 x [1024 x i32] addrspace(3)*> %array, <4 x i16> zeroinitializer, <4 x i16> <i16 16, i16 16, i16 16, i16 16>
  %p0 = extractelement <4 x i32 addrspace(3)*> %p, i32 0
  %p1 = extractelement <4 x i32 addrspace(3)*> %p, i32 1
  %p2 = extractelement <4 x i32 addrspace(3)*> %p, i32 2
  %p3 = extractelement <4 x i32 addrspace(3)*> %p, i32 3
  store i32 99, i32 addrspace(3)* %p0
  store i32 99, i32 addrspace(3)* %p1
  store i32 99, i32 addrspace(3)* %p2
  store i32 99, i32 addrspace(3)* %p3
  ret void
}

; CHECK-LABEL: {{^}}gep_as_vector_v2:
; SI: s_add_i32
; SI: s_add_i32
; CHECK-DAG: v_mov_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}
; CHECK-DAG: v_mov_b32_e32 {{v[0-9]+}}, {{s[0-9]+}}
; CI-DAG: ds_write_b32 v{{[0-9]+}}, v{{[0-9]+}} offset:64
; CI-DAG: ds_write_b32 v{{[0-9]+}}, v{{[0-9]+}} offset:64
; CHECK: s_endpgm
define amdgpu_kernel void @gep_as_vector_v2(<2 x [1024 x i32] addrspace(3)*> %array) nounwind {
  %p = getelementptr [1024 x i32], <2 x [1024 x i32] addrspace(3)*> %array, <2 x i16> zeroinitializer, <2 x i16> <i16 16, i16 16>
  %p0 = extractelement <2 x i32 addrspace(3)*> %p, i32 0
  %p1 = extractelement <2 x i32 addrspace(3)*> %p, i32 1
  store i32 99, i32 addrspace(3)* %p0
  store i32 99, i32 addrspace(3)* %p1
  ret void
}

