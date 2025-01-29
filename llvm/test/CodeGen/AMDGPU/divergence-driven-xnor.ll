; RUN: llc -mtriple=amdgcn -stop-after=amdgpu-isel < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx906 -stop-after=amdgpu-isel < %s | FileCheck -check-prefix=GCN_DL %s
; RUN: llc -mtriple=amdgcn -enable-new-pm -stop-after=amdgpu-isel < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx906 -enable-new-pm -stop-after=amdgpu-isel < %s | FileCheck -check-prefix=GCN_DL %s

; GCN-LABEL: name:            uniform_xnor_i64
; GCN: S_XNOR_B64
define amdgpu_kernel void @uniform_xnor_i64(ptr addrspace(1) %out, i64 %a, i64 %b) {
  %xor = xor i64 %a, %b
  %res = xor i64 %xor, -1
  store i64 %res, ptr addrspace(1) %out
  ret void
}
; GCN-LABEL: name:            divergent_xnor_i64
; GCN: V_XOR_B32_e64
; GCN: V_XOR_B32_e64
; GCN: V_NOT_B32_e32
; GCN: V_NOT_B32_e32
; GCN_DL: V_XNOR_B32_e64
; GCN_DL: V_XNOR_B32_e64
define i64 @divergent_xnor_i64(ptr addrspace(1) %out, i64 %a, i64 %b) {
  %xor = xor i64 %a, %b
  %res = xor i64 %xor, -1
  ret i64 %res
}

; GCN-LABEL: name:            uniform_xnor_i32
; GCN: S_XNOR_B32
define amdgpu_kernel void @uniform_xnor_i32(ptr addrspace(1) %out, i32 %a, i32 %b) {
  %xor = xor i32 %a, %b
  %res = xor i32 %xor, -1
  store i32 %res, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: name:            divergent_xnor_i32
; GCN: V_XOR_B32_e64
; GCN: V_NOT_B32_e32
; GCN_DL: V_XNOR_B32_e64
define i32 @divergent_xnor_i32(ptr addrspace(1) %out, i32 %a, i32 %b) {
  %xor = xor i32 %a, %b
  %res = xor i32 %xor, -1
  ret i32 %res
}

declare i32 @llvm.amdgcn.workitem.id.x() #0
