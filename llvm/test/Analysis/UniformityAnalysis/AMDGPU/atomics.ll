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

attributes #0 = { nounwind }
attributes #1 = { argmemonly nounwind willreturn }
