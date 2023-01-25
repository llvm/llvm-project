; RUN: llc -march=amdgcn -mcpu=gfx600 -verify-machineinstrs < %s | FileCheck --check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=gfx700 -verify-machineinstrs < %s | FileCheck --check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=gfx801 -verify-machineinstrs < %s | FileCheck --check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck --check-prefix=GCN %s

; GCN-LABEL: {{^}}scalar_andn2_i32_one_use
; GCN: s_andn2_b32
define amdgpu_kernel void @scalar_andn2_i32_one_use(
    ptr addrspace(1) %r0, i32 %a, i32 %b) {
entry:
  %nb = xor i32 %b, -1
  %r0.val = and i32 %a, %nb
  store i32 %r0.val, ptr addrspace(1) %r0
  ret void
}

; GCN-LABEL: {{^}}scalar_andn2_i64_one_use
; GCN: s_andn2_b64
define amdgpu_kernel void @scalar_andn2_i64_one_use(
    ptr addrspace(1) %r0, i64 %a, i64 %b) {
entry:
  %nb = xor i64 %b, -1
  %r0.val = and i64 %a, %nb
  store i64 %r0.val, ptr addrspace(1) %r0
  ret void
}

; GCN-LABEL: {{^}}scalar_orn2_i32_one_use
; GCN: s_orn2_b32
define amdgpu_kernel void @scalar_orn2_i32_one_use(
    ptr addrspace(1) %r0, i32 %a, i32 %b) {
entry:
  %nb = xor i32 %b, -1
  %r0.val = or i32 %a, %nb
  store i32 %r0.val, ptr addrspace(1) %r0
  ret void
}

; GCN-LABEL: {{^}}scalar_orn2_i64_one_use
; GCN: s_orn2_b64
define amdgpu_kernel void @scalar_orn2_i64_one_use(
    ptr addrspace(1) %r0, i64 %a, i64 %b) {
entry:
  %nb = xor i64 %b, -1
  %r0.val = or i64 %a, %nb
  store i64 %r0.val, ptr addrspace(1) %r0
  ret void
}

; GCN-LABEL: {{^}}vector_andn2_i32_s_v_one_use
; GCN: v_not_b32
; GCN: v_and_b32
define amdgpu_kernel void @vector_andn2_i32_s_v_one_use(
    ptr addrspace(1) %r0, i32 %s) {
entry:
  %v = call i32 @llvm.amdgcn.workitem.id.x() #1
  %not = xor i32 %v, -1
  %r0.val = and i32 %s, %not
  store i32 %r0.val, ptr addrspace(1) %r0
  ret void
}

; GCN-LABEL: {{^}}vector_andn2_i32_v_s_one_use
; GCN: s_not_b32
; GCN: v_and_b32
define amdgpu_kernel void @vector_andn2_i32_v_s_one_use(
    ptr addrspace(1) %r0, i32 %s) {
entry:
  %v = call i32 @llvm.amdgcn.workitem.id.x() #1
  %not = xor i32 %s, -1
  %r0.val = and i32 %v, %not
  store i32 %r0.val, ptr addrspace(1) %r0
  ret void
}

; GCN-LABEL: {{^}}vector_orn2_i32_s_v_one_use
; GCN: v_not_b32
; GCN: v_or_b32
define amdgpu_kernel void @vector_orn2_i32_s_v_one_use(
    ptr addrspace(1) %r0, i32 %s) {
entry:
  %v = call i32 @llvm.amdgcn.workitem.id.x() #1
  %not = xor i32 %v, -1
  %r0.val = or i32 %s, %not
  store i32 %r0.val, ptr addrspace(1) %r0
  ret void
}

; GCN-LABEL: {{^}}vector_orn2_i32_v_s_one_use
; GCN: s_not_b32
; GCN: v_or_b32
define amdgpu_kernel void @vector_orn2_i32_v_s_one_use(
    ptr addrspace(1) %r0, i32 %s) {
entry:
  %v = call i32 @llvm.amdgcn.workitem.id.x() #1
  %not = xor i32 %s, -1
  %r0.val = or i32 %v, %not
  store i32 %r0.val, ptr addrspace(1) %r0
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @llvm.amdgcn.workitem.id.x() #0
