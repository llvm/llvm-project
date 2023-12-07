; RUN: opt -mtriple amdgcn-- -passes='print<uniformity>' -disable-output %s 2>&1 | FileCheck %s

; CHECK: DIVERGENT: %orig = atomicrmw xchg ptr %ptr, i32 %val seq_cst
define amdgpu_kernel void @test1(ptr %ptr, i32 %val) #0 {
  %orig = atomicrmw xchg ptr %ptr, i32 %val seq_cst
  store i32 %orig, ptr %ptr
  ret void
}

; CHECK: DIVERGENT: %orig = cmpxchg ptr %ptr, i32 %cmp, i32 %new seq_cst seq_cst
define amdgpu_kernel void @test2(ptr %ptr, i32 %cmp, i32 %new) {
  %orig = cmpxchg ptr %ptr, i32 %cmp, i32 %new seq_cst seq_cst
  %val = extractvalue { i32, i1 } %orig, 0
  store i32 %val, ptr %ptr
  ret void
}

; CHECK: DIVERGENT: %ret = call i32 @llvm.amdgcn.global.atomic.csub.p1(ptr addrspace(1) %ptr, i32 %val)
define amdgpu_kernel void @test_atomic_csub_i32(ptr addrspace(1) %ptr, i32 %val) #0 {
  %ret = call i32 @llvm.amdgcn.global.atomic.csub.p1(ptr addrspace(1) %ptr, i32 %val)
  store i32 %ret, i32 addrspace(1)* %ptr, align 4
  ret void
}

declare i32 @llvm.amdgcn.global.atomic.csub.p1(ptr addrspace(1) nocapture, i32) #1

attributes #0 = { nounwind }
attributes #1 = { argmemonly nounwind willreturn }
