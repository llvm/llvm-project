; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s 2>&1 | FileCheck %s

; CHECK: error: <unknown>:0:0: in function ubfe_i64 void (ptr addrspace(1), i64): llvm.amdgcn.ubfe only supports i32

define amdgpu_kernel void @ubfe_i64(ptr addrspace(1) %out, i64 %src) {
  %bfe = call i64 @llvm.amdgcn.ubfe.i64(i64 %src, i32 8, i32 16)
  store i64 %bfe, ptr addrspace(1) %out
  ret void
}

declare i64 @llvm.amdgcn.ubfe.i64(i64, i32, i32)
