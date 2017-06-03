; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=hawaii -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=fiji -verify-machineinstrs < %s | FileCheck %s

declare i32 @llvm.amdgcn.ds.swizzle(i32, i32) #0

; FUNC-LABEL: {{^}}ds_swizzle:
; CHECK: ds_swizzle_b32 v{{[0-9]+}}, v{{[0-9]+}} offset:swizzle(BITMASK_PERM,"00p11")
define amdgpu_kernel void @ds_swizzle(i32 addrspace(1)* %out, i32 %src) nounwind {
  %swizzle = call i32 @llvm.amdgcn.ds.swizzle(i32 %src, i32 100) #0
  store i32 %swizzle, i32 addrspace(1)* %out, align 4
  ret void
}

attributes #0 = { nounwind readnone convergent }
