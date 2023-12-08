; RUN: opt -S -mtriple=amdgcn-unknown-unknown -passes=amdgpu-printf-runtime-binding < %s 2>&1 | FileCheck %s

@.str = private unnamed_addr addrspace(4) constant [6 x i8] c"%s:%d\00", align 1

define amdgpu_kernel void @test_kernel(i32 %n) {
entry:
  %str = alloca [9 x i8], align 1, addrspace(5)
  %call1 = call i32 (ptr addrspace(4), ...) @printf(ptr addrspace(4) @.str, ptr addrspace(5) %str, i32 %n)
  %call2 = call <2 x i64> (ptr, i32, i64, i64, i64, i64, i64, i64, i64, i64) @__ockl_hostcall_internal(ptr undef, i32 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i64 9)
  ret void
}

declare i32 @printf(ptr addrspace(4), ...)

declare <2 x i64> @__ockl_hostcall_internal(ptr, i32, i64, i64, i64, i64, i64, i64, i64, i64)

; CHECK-NOT: error:
; CHECK-NOT: warning:
