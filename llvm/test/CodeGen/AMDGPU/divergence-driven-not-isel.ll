; RUN: llc -mtriple=amdgcn -stop-after=amdgpu-isel < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn -enable-new-pm -stop-after=amdgpu-isel < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: name:            scalar_not_i32
; GCN: S_NOT_B32
define amdgpu_kernel void @scalar_not_i32(ptr addrspace(1) %out, i32 %val) {
  %not.val = xor i32 %val, -1
  store i32 %not.val, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: name:            scalar_not_i64
; GCN: S_NOT_B64
define amdgpu_kernel void @scalar_not_i64(ptr addrspace(1) %out, i64 %val) {
  %not.val = xor i64 %val, -1
  store i64 %not.val, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: name:            vector_not_i32
; GCN: V_NOT_B32_e32
define i32 @vector_not_i32(i32 %val) {
  %not.val = xor i32 %val, -1
  ret i32 %not.val
}

; GCN-LABEL: name:            vector_not_i64
; GCN: V_NOT_B32_e32
; GCN: V_NOT_B32_e32
define i64 @vector_not_i64(i64 %val) {
  %not.val = xor i64 %val, -1
  ret i64 %not.val
}


