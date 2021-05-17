; RUN: opt -mtriple=amdgcn-- -enable-new-pm=0 -analyze -divergence -use-gpu-divergence-analysis %s | FileCheck %s
; RUN: opt -mtriple amdgcn-- -passes='print<divergence>' -disable-output %s 2>&1 | FileCheck %s

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

; CHECK: DIVERGENT: %ret = call i32 @llvm.amdgcn.atomic.inc.i32.p1i32(i32 addrspace(1)* %ptr, i32 %val, i32 0, i32 0, i1 false)
define i32 @test_atomic_inc_i32(i32 addrspace(1)* %ptr, i32 %val) #0 {
  %ret = call i32 @llvm.amdgcn.atomic.inc.i32.p1i32(i32 addrspace(1)* %ptr, i32 %val, i32 0, i32 0, i1 false)
  ret i32 %ret
}

; CHECK: DIVERGENT: %ret = call i64 @llvm.amdgcn.atomic.inc.i64.p1i64(i64 addrspace(1)* %ptr, i64 %val, i32 0, i32 0, i1 false)
define i64 @test_atomic_inc_i64(i64 addrspace(1)* %ptr, i64 %val) #0 {
  %ret = call i64 @llvm.amdgcn.atomic.inc.i64.p1i64(i64 addrspace(1)* %ptr, i64 %val, i32 0, i32 0, i1 false)
  ret i64 %ret
}

; CHECK: DIVERGENT: %ret = call i32 @llvm.amdgcn.atomic.dec.i32.p1i32(i32 addrspace(1)* %ptr, i32 %val, i32 0, i32 0, i1 false)
define i32 @test_atomic_dec_i32(i32 addrspace(1)* %ptr, i32 %val) #0 {
  %ret = call i32 @llvm.amdgcn.atomic.dec.i32.p1i32(i32 addrspace(1)* %ptr, i32 %val, i32 0, i32 0, i1 false)
  ret i32 %ret
}

; CHECK: DIVERGENT: %ret = call i64 @llvm.amdgcn.atomic.dec.i64.p1i64(i64 addrspace(1)* %ptr, i64 %val, i32 0, i32 0, i1 false)
define i64 @test_atomic_dec_i64(i64 addrspace(1)* %ptr, i64 %val) #0 {
  %ret = call i64 @llvm.amdgcn.atomic.dec.i64.p1i64(i64 addrspace(1)* %ptr, i64 %val, i32 0, i32 0, i1 false)
  ret i64 %ret
}

declare i32 @llvm.amdgcn.atomic.inc.i32.p1i32(i32 addrspace(1)* nocapture, i32, i32, i32, i1) #1
declare i64 @llvm.amdgcn.atomic.inc.i64.p1i64(i64 addrspace(1)* nocapture, i64, i32, i32, i1) #1
declare i32 @llvm.amdgcn.atomic.dec.i32.p1i32(i32 addrspace(1)* nocapture, i32, i32, i32, i1) #1
declare i64 @llvm.amdgcn.atomic.dec.i64.p1i64(i64 addrspace(1)* nocapture, i64, i32, i32, i1) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind argmemonly }
