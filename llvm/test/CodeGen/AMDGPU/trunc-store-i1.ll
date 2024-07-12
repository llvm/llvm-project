; RUN: llc -mtriple=amdgcn -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs< %s | FileCheck -enable-var-scope --check-prefix=GCN %s


; GCN-LABEL: {{^}}global_truncstore_i32_to_i1:
; GCN: s_load_dword [[LOAD:s[0-9]+]],
; GCN: s_and_b32 [[SREG:s[0-9]+]], [[LOAD]], 1
; GCN: v_mov_b32_e32 [[VREG:v[0-9]+]], [[SREG]]
; GCN: buffer_store_byte [[VREG]],
define amdgpu_kernel void @global_truncstore_i32_to_i1(ptr addrspace(1) %out, i32 %val) nounwind {
  %trunc = trunc i32 %val to i1
  store i1 %trunc, ptr addrspace(1) %out, align 1
  ret void
}

; GCN-LABEL: {{^}}global_truncstore_i64_to_i1:
; GCN: buffer_store_byte
define amdgpu_kernel void @global_truncstore_i64_to_i1(ptr addrspace(1) %out, i64 %val) nounwind {
  %trunc = trunc i64 %val to i1
  store i1 %trunc, ptr addrspace(1) %out, align 1
  ret void
}

; FIXME: VGPR on VI
; GCN-LABEL: {{^}}s_arg_global_truncstore_i16_to_i1:
; GCN: s_load_dword [[LOAD:s[0-9]+]],
; GCN: s_and_b32 [[SREG:s[0-9]+]], [[LOAD]], 1
; GCN: v_mov_b32_e32 [[VREG:v[0-9]+]], [[SREG]]
; GCN: buffer_store_byte [[VREG]],
define amdgpu_kernel void @s_arg_global_truncstore_i16_to_i1(ptr addrspace(1) %out, i16 %val) nounwind {
  %trunc = trunc i16 %val to i1
  store i1 %trunc, ptr addrspace(1) %out, align 1
  ret void
}
; GCN-LABEL: {{^}}global_truncstore_i16_to_i1:
define amdgpu_kernel void @global_truncstore_i16_to_i1(ptr addrspace(1) %out, i16 %val0, i16 %val1) nounwind {
  %add = add i16 %val0, %val1
  %trunc = trunc i16 %add to i1
  store i1 %trunc, ptr addrspace(1) %out, align 1
  ret void
}
