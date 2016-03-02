; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=fiji -verify-machineinstrs < %s | FileCheck %s

declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #0

; FUNC-LABEL: {{^}}ds_bpermute:
; CHECK: ds_bpermute_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define void @ds_bpermute(i32 addrspace(1)* %out, i32 %index, i32 %src) nounwind {
  %bpermute = call i32 @llvm.amdgcn.ds.bpermute(i32 %index, i32 %src) #0
  store i32 %bpermute, i32 addrspace(1)* %out, align 4
  ret void
}

attributes #0 = { nounwind readnone convergent }
