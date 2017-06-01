; RUN: opt -mtriple=amdgcn-- -analyze -divergence %s | FileCheck %s

; CHECK: DIVERGENT: %orig = atomicrmw xchg i32* %ptr, i32 %val seq_cst
define i32 @test1(i32* %ptr, i32 %val) #0 {
  %orig = atomicrmw xchg i32* %ptr, i32 %val seq_cst
  ret i32 %orig
}

; CHECK: DIVERGENT: %orig = cmpxchg i32* %ptr, i32 %cmp, i32 %new seq_cst seq_cst
define {i32, i1} @test2(i32* %ptr, i32 %cmp, i32 %new) {
  %orig = cmpxchg i32* %ptr, i32 %cmp, i32 %new seq_cst seq_cst
  ret {i32, i1} %orig
}

attributes #0 = { nounwind }
