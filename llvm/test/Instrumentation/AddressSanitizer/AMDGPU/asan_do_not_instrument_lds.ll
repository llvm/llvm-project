; RUN: opt < %s -passes=asan -S | FileCheck %s
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
target triple = "amdgcn-amd-amdhsa"

; Memory access to lds are not instrumented

@count = addrspace(3) global [100 x i32] undef, align 16

define protected amdgpu_kernel void @lds_store(i32 %i) sanitize_address {
entry:
  ; CHECK-NOT: call * __asan_report
  %arrayidx1 = getelementptr inbounds [100 x i32], [100 x i32] addrspace(3)* @count, i32 0, i32 %i
  store i32 0, i32 addrspace(3)* %arrayidx1, align 4
  ret void
}

define protected amdgpu_kernel void @lds_load(i32 %i) sanitize_address {
entry:
  ; CHECK-NOT: call * __asan_report
  %arrayidx1 = getelementptr inbounds [100 x i32], [100 x i32] addrspace(3)* @count, i32 0, i32 %i
  %0 = load i32, i32 addrspace(3)* %arrayidx1, align 4
  ret void
}
