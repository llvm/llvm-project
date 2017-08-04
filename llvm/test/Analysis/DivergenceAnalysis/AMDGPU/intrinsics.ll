; RUN: opt -mtriple=amdgcn-- -analyze -divergence %s | FileCheck %s

; CHECK: DIVERGENT: %swizzle = call i32 @llvm.amdgcn.ds.swizzle(i32 %src, i32 100) #0
define amdgpu_kernel void @ds_swizzle(i32 addrspace(1)* %out, i32 %src) #0 {
  %swizzle = call i32 @llvm.amdgcn.ds.swizzle(i32 %src, i32 100) #0
  store i32 %swizzle, i32 addrspace(1)* %out, align 4
  ret void
}

declare i32 @llvm.amdgcn.ds.swizzle(i32, i32) #1

attributes #0 = { nounwind convergent }
attributes #1 = { nounwind readnone convergent }
