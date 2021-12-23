; RUN: llc -march=amdgcn -stop-after=amdgpu-isel < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: name:            uniform_trunc_i16_to_i1
; GCN: S_AND_B32 1
; GCN: S_CMP_EQ_U32
define amdgpu_kernel void @uniform_trunc_i16_to_i1(i1 addrspace(1)* %out, i16 %x, i1 %z) {
  %setcc = icmp slt i16 %x, 0
  %select = select i1 %setcc, i1 true, i1 %z
  store i1 %select, i1 addrspace(1)* %out
  ret void
}

; GCN-LABEL: name:            divergent_trunc_i16_to_i1
; GCN: V_AND_B32_e64 1
; GCN: V_CMP_EQ_U32_e64
define i1 @divergent_trunc_i16_to_i1(i1 addrspace(1)* %out, i16 %x, i1 %z) {
  %setcc = icmp slt i16 %x, 0
  %select = select i1 %setcc, i1 true, i1 %z
  ret i1 %select
}

; GCN-LABEL: name:            uniform_trunc_i32_to_i1
; GCN: S_AND_B32 1
; GCN: S_CMP_EQ_U32
define amdgpu_kernel void @uniform_trunc_i32_to_i1(i1 addrspace(1)* %out, i32 %x, i1 %z) {
  %setcc = icmp slt i32 %x, 0
  %select = select i1 %setcc, i1 true, i1 %z
  store i1 %select, i1 addrspace(1)* %out
  ret void
}

; GCN-LABEL: name:            divergent_trunc_i32_to_i1
; GCN: V_AND_B32_e64 1
; GCN: V_CMP_EQ_U32_e64
define i1 @divergent_trunc_i32_to_i1(i1 addrspace(1)* %out, i32 %x, i1 %z) {
  %setcc = icmp slt i32 %x, 0
  %select = select i1 %setcc, i1 true, i1 %z
  ret i1 %select
}

; GCN-LABEL: name:            uniform_trunc_i64_to_i1
; GCN: S_AND_B32 1
; GCN: S_CMP_EQ_U32
define amdgpu_kernel void @uniform_trunc_i64_to_i1(i1 addrspace(1)* %out, i64 %x, i1 %z) {
  %setcc = icmp slt i64 %x, 0
  %select = select i1 %setcc, i1 true, i1 %z
  store i1 %select, i1 addrspace(1)* %out
  ret void
}

; GCN-LABEL: name:            divergent_trunc_i64_to_i1
; GCN: V_AND_B32_e64 1
; GCN: V_CMP_EQ_U32_e64
define i1 @divergent_trunc_i64_to_i1(i1 addrspace(1)* %out, i64 %x, i1 %z) {
  %setcc = icmp slt i64 %x, 0
  %select = select i1 %setcc, i1 true, i1 %z
  ret i1 %select
}

