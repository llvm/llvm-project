; RUN: llc -march=amdgcn -verify-machineinstrs< %s | FileCheck --check-prefix=SI --check-prefix=BOTH %s
; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs< %s | FileCheck --check-prefix=CI --check-prefix=BOTH %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs< %s | FileCheck --check-prefix=CI --check-prefix=BOTH %s

; BOTH-LABEL: {{^}}local_i32_load
; BOTH: ds_read_b32 [[REG:v[0-9]+]], v{{[0-9]+}} offset:28
; BOTH: buffer_store_dword [[REG]],
define amdgpu_kernel void @local_i32_load(i32 addrspace(1)* %out, i32 addrspace(3)* %in) nounwind {
  %gep = getelementptr i32, i32 addrspace(3)* %in, i32 7
  %val = load i32, i32 addrspace(3)* %gep, align 4
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; BOTH-LABEL: {{^}}local_i32_load_0_offset
; BOTH: ds_read_b32 [[REG:v[0-9]+]], v{{[0-9]+}}
; BOTH: buffer_store_dword [[REG]],
define amdgpu_kernel void @local_i32_load_0_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %in) nounwind {
  %val = load i32, i32 addrspace(3)* %in, align 4
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; BOTH-LABEL: {{^}}local_i8_load_i16_max_offset:
; BOTH-NOT: ADD
; BOTH: ds_read_u8 [[REG:v[0-9]+]], {{v[0-9]+}} offset:65535
; BOTH: buffer_store_byte [[REG]],
define amdgpu_kernel void @local_i8_load_i16_max_offset(i8 addrspace(1)* %out, i8 addrspace(3)* %in) nounwind {
  %gep = getelementptr i8, i8 addrspace(3)* %in, i32 65535
  %val = load i8, i8 addrspace(3)* %gep, align 4
  store i8 %val, i8 addrspace(1)* %out, align 4
  ret void
}

; BOTH-LABEL: {{^}}local_i8_load_over_i16_max_offset:
; The LDS offset will be 65536 bytes, which is larger than the size of LDS on
; SI, which is why it is being OR'd with the base pointer.
; SI: s_or_b32 [[ADDR:s[0-9]+]], s{{[0-9]+}}, 0x10000
; CI: s_add_i32 [[ADDR:s[0-9]+]], s{{[0-9]+}}, 0x10000
; BOTH: v_mov_b32_e32 [[VREGADDR:v[0-9]+]], [[ADDR]]
; BOTH: ds_read_u8 [[REG:v[0-9]+]], [[VREGADDR]]
; BOTH: buffer_store_byte [[REG]],
define amdgpu_kernel void @local_i8_load_over_i16_max_offset(i8 addrspace(1)* %out, i8 addrspace(3)* %in) nounwind {
  %gep = getelementptr i8, i8 addrspace(3)* %in, i32 65536
  %val = load i8, i8 addrspace(3)* %gep, align 4
  store i8 %val, i8 addrspace(1)* %out, align 4
  ret void
}

; BOTH-LABEL: {{^}}local_i64_load:
; BOTH-NOT: ADD
; BOTH: ds_read_b64 [[REG:v[[0-9]+:[0-9]+]]], v{{[0-9]+}} offset:56
; BOTH: buffer_store_dwordx2 [[REG]],
define amdgpu_kernel void @local_i64_load(i64 addrspace(1)* %out, i64 addrspace(3)* %in) nounwind {
  %gep = getelementptr i64, i64 addrspace(3)* %in, i32 7
  %val = load i64, i64 addrspace(3)* %gep, align 8
  store i64 %val, i64 addrspace(1)* %out, align 8
  ret void
}

; BOTH-LABEL: {{^}}local_i64_load_0_offset
; BOTH: ds_read_b64 [[REG:v\[[0-9]+:[0-9]+\]]], v{{[0-9]+}}
; BOTH: buffer_store_dwordx2 [[REG]],
define amdgpu_kernel void @local_i64_load_0_offset(i64 addrspace(1)* %out, i64 addrspace(3)* %in) nounwind {
  %val = load i64, i64 addrspace(3)* %in, align 8
  store i64 %val, i64 addrspace(1)* %out, align 8
  ret void
}

; BOTH-LABEL: {{^}}local_f64_load:
; BOTH-NOT: ADD
; BOTH: ds_read_b64 [[REG:v[[0-9]+:[0-9]+]]], v{{[0-9]+}} offset:56
; BOTH: buffer_store_dwordx2 [[REG]],
define amdgpu_kernel void @local_f64_load(double addrspace(1)* %out, double addrspace(3)* %in) nounwind {
  %gep = getelementptr double, double addrspace(3)* %in, i32 7
  %val = load double, double addrspace(3)* %gep, align 8
  store double %val, double addrspace(1)* %out, align 8
  ret void
}

; BOTH-LABEL: {{^}}local_f64_load_0_offset
; BOTH: ds_read_b64 [[REG:v\[[0-9]+:[0-9]+\]]], v{{[0-9]+}}
; BOTH: buffer_store_dwordx2 [[REG]],
define amdgpu_kernel void @local_f64_load_0_offset(double addrspace(1)* %out, double addrspace(3)* %in) nounwind {
  %val = load double, double addrspace(3)* %in, align 8
  store double %val, double addrspace(1)* %out, align 8
  ret void
}

; BOTH-LABEL: {{^}}local_i64_store:
; BOTH-NOT: ADD
; BOTH: ds_write_b64 v{{[0-9]+}}, {{v\[[0-9]+:[0-9]+\]}} offset:56
define amdgpu_kernel void @local_i64_store(i64 addrspace(3)* %out) nounwind {
  %gep = getelementptr i64, i64 addrspace(3)* %out, i32 7
  store i64 5678, i64 addrspace(3)* %gep, align 8
  ret void
}

; BOTH-LABEL: {{^}}local_i64_store_0_offset:
; BOTH-NOT: ADD
; BOTH: ds_write_b64 v{{[0-9]+}}, {{v\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @local_i64_store_0_offset(i64 addrspace(3)* %out) nounwind {
  store i64 1234, i64 addrspace(3)* %out, align 8
  ret void
}

; BOTH-LABEL: {{^}}local_f64_store:
; BOTH-NOT: ADD
; BOTH: ds_write_b64 v{{[0-9]+}}, {{v\[[0-9]+:[0-9]+\]}} offset:56
define amdgpu_kernel void @local_f64_store(double addrspace(3)* %out) nounwind {
  %gep = getelementptr double, double addrspace(3)* %out, i32 7
  store double 16.0, double addrspace(3)* %gep, align 8
  ret void
}

; BOTH-LABEL: {{^}}local_f64_store_0_offset
; BOTH: ds_write_b64 v{{[0-9]+}}, {{v\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @local_f64_store_0_offset(double addrspace(3)* %out) nounwind {
  store double 20.0, double addrspace(3)* %out, align 8
  ret void
}

; BOTH-LABEL: {{^}}local_v2i64_store:
; BOTH-NOT: ADD
; BOTH: ds_write2_b64 v{{[0-9]+}}, {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}} offset0:14 offset1:15
; BOTH: s_endpgm
define amdgpu_kernel void @local_v2i64_store(<2 x i64> addrspace(3)* %out) nounwind {
  %gep = getelementptr <2 x i64>, <2 x i64> addrspace(3)* %out, i32 7
  store <2 x i64> <i64 5678, i64 5678>, <2 x i64> addrspace(3)* %gep, align 16
  ret void
}

; BOTH-LABEL: {{^}}local_v2i64_store_0_offset:
; BOTH-NOT: ADD
; BOTH: ds_write2_b64 v{{[0-9]+}}, {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}} offset1:1
; BOTH: s_endpgm
define amdgpu_kernel void @local_v2i64_store_0_offset(<2 x i64> addrspace(3)* %out) nounwind {
  store <2 x i64> <i64 1234, i64 1234>, <2 x i64> addrspace(3)* %out, align 16
  ret void
}

; BOTH-LABEL: {{^}}local_v4i64_store:
; BOTH-NOT: ADD
; BOTH-DAG: ds_write2_b64 v{{[0-9]+}}, {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}} offset0:30 offset1:31
; BOTH-DAG: ds_write2_b64 v{{[0-9]+}}, {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}} offset0:28 offset1:29
; BOTH: s_endpgm
define amdgpu_kernel void @local_v4i64_store(<4 x i64> addrspace(3)* %out) nounwind {
  %gep = getelementptr <4 x i64>, <4 x i64> addrspace(3)* %out, i32 7
  store <4 x i64> <i64 5678, i64 5678, i64 5678, i64 5678>, <4 x i64> addrspace(3)* %gep, align 16
  ret void
}

; BOTH-LABEL: {{^}}local_v4i64_store_0_offset:
; BOTH-NOT: ADD
; BOTH-DAG: ds_write2_b64 v{{[0-9]+}}, {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}} offset0:2 offset1:3
; BOTH-DAG: ds_write2_b64 v{{[0-9]+}}, {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}} offset1:1
; BOTH: s_endpgm
define amdgpu_kernel void @local_v4i64_store_0_offset(<4 x i64> addrspace(3)* %out) nounwind {
  store <4 x i64> <i64 1234, i64 1234, i64 1234, i64 1234>, <4 x i64> addrspace(3)* %out, align 16
  ret void
}
