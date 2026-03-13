; RUN: opt < %s -passes=asan -S | FileCheck %s
target triple = "amdgcn-amd-amdhsa"

; Memory access to lds are not instrumented

@count = addrspace(3) global [100 x i32] undef, align 16

define protected amdgpu_kernel void @lds_store(i32 %i) sanitize_address {
entry:
  ; CHECK-LABEL: @lds_store(
  ; CHECK-NOT: call {{[a-zA-Z]}}
  %arrayidx1 = getelementptr inbounds [100 x i32], ptr addrspace(3) @count, i32 0, i32 %i
  store i32 0, ptr addrspace(3) %arrayidx1, align 4
  ret void
}

define protected amdgpu_kernel void @lds_load(i32 %i) sanitize_address {
entry:
  ; CHECK-LABEL: @lds_load(
  ; CHECK-NOT: call {{[a-zA-Z]}}
  %arrayidx1 = getelementptr inbounds [100 x i32], ptr addrspace(3) @count, i32 0, i32 %i
  %0 = load i32, ptr addrspace(3) %arrayidx1, align 4
  ret void
}

; CHECK-LABEL: define internal void @asan.module_ctor()
