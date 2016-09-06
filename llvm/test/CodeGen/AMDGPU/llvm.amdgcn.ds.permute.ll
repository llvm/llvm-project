; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=fiji -verify-machineinstrs < %s | FileCheck %s

declare i32 @llvm.amdgcn.ds.permute(i32, i32) #0

; CHECK-LABEL: {{^}}ds_permute:
; CHECK: ds_permute_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; CHECK: s_waitcnt lgkmcnt
define void @ds_permute(i32 addrspace(1)* %out, i32 %index, i32 %src) nounwind {
  %bpermute = call i32 @llvm.amdgcn.ds.permute(i32 %index, i32 %src) #0
  store i32 %bpermute, i32 addrspace(1)* %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}ds_permute_imm_offset:
; CHECK: ds_permute_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:4
; CHECK: s_waitcnt lgkmcnt
define void @ds_permute_imm_offset(i32 addrspace(1)* %out, i32 %base_index, i32 %src) nounwind {
  %index = add i32 %base_index, 4
  %bpermute = call i32 @llvm.amdgcn.ds.permute(i32 %index, i32 %src) #0
  store i32 %bpermute, i32 addrspace(1)* %out, align 4
  ret void
}

attributes #0 = { nounwind readnone convergent }
