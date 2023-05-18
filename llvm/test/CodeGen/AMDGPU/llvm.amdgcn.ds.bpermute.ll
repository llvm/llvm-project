; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=fiji -verify-machineinstrs < %s | FileCheck %s

declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #0

; CHECK-LABEL: {{^}}ds_bpermute:
; CHECK: ds_bpermute_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @ds_bpermute(ptr addrspace(1) %out, i32 %index, i32 %src) nounwind {
  %bpermute = call i32 @llvm.amdgcn.ds.bpermute(i32 %index, i32 %src) #0
  store i32 %bpermute, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}ds_bpermute_imm_offset:
; CHECK: ds_bpermute_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:4
define amdgpu_kernel void @ds_bpermute_imm_offset(ptr addrspace(1) %out, i32 %base_index, i32 %src) nounwind {
  %index = add i32 %base_index, 4
  %bpermute = call i32 @llvm.amdgcn.ds.bpermute(i32 %index, i32 %src) #0
  store i32 %bpermute, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}ds_bpermute_imm_index:
; CHECK: ds_bpermute_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:64
define amdgpu_kernel void @ds_bpermute_imm_index(ptr addrspace(1) %out, i32 %base_index, i32 %src) nounwind {
  %bpermute = call i32 @llvm.amdgcn.ds.bpermute(i32 64, i32 %src) #0
  store i32 %bpermute, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}ds_bpermute_add_shl:
; CHECK: ds_bpermute_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:4
; CHECK: s_waitcnt lgkmcnt
define void @ds_bpermute_add_shl(ptr addrspace(1) %out, i32 %base_index, i32 %src) nounwind {
  %index = add i32 %base_index, 1
  %byte_index = shl i32 %index, 2
  %bpermute = call i32 @llvm.amdgcn.ds.bpermute(i32 %byte_index, i32 %src) #0
  store i32 %bpermute, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}ds_bpermute_or_shl:
; CHECK: ds_bpermute_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:4
; CHECK: s_waitcnt lgkmcnt
define void @ds_bpermute_or_shl(ptr addrspace(1) %out, i32 %base_index, i32 %src) nounwind {
  %masked = and i32 %base_index, 62
  %index = or i32 %masked, 1
  %byte_index = shl i32 %index, 2
  %bpermute = call i32 @llvm.amdgcn.ds.bpermute(i32 %byte_index, i32 %src) #0
  store i32 %bpermute, ptr addrspace(1) %out, align 4
  ret void
}

attributes #0 = { nounwind readnone convergent }
