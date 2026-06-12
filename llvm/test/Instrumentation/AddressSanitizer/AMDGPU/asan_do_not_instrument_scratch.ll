; RUN: opt < %s -passes=asan -S | FileCheck %s
target triple = "amdgcn-amd-amdhsa"

; Memory access to scratch are not instrumented

define protected amdgpu_kernel void @scratch_store(i32 %i) sanitize_address {
entry:
  ; CHECK-NOT: call * __asan_report
  %c = alloca i32, align 4, addrspace(5)
  store i32 0, ptr addrspace(5) %c, align 4
  ret void
}

define protected amdgpu_kernel void @scratch_load(i32 %i) sanitize_address {
entry:
  ; CHECK-NOT: call * __asan_report
  %c = alloca i32, align 4, addrspace(5)
  %0 = load i32, ptr addrspace(5) %c, align 4
  ret void
}
