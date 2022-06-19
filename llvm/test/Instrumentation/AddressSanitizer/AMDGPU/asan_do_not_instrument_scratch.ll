; RUN: opt < %s -passes='asan-pipeline' -S | FileCheck %s
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
target triple = "amdgcn-amd-amdhsa"

; Memory access to scratch are not instrumented

define protected amdgpu_kernel void @scratch_store(i32 %i) sanitize_address {
entry:
  ; CHECK-NOT: call * __asan_report
  %c = alloca i32, align 4, addrspace(5)
  store i32 0, i32 addrspace(5)* %c, align 4
  ret void
}

define protected amdgpu_kernel void @scratch_load(i32 %i) sanitize_address {
entry:
  ; CHECK-NOT: call * __asan_report
  %c = alloca i32, align 4, addrspace(5)
  %0 = load i32, i32 addrspace(5)* %c, align 4
  ret void
}
