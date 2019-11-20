; RUN: not opt -S -mtriple=amdgcn-unknown-unknown -amdgpu-printf-runtime-binding < %s 2>&1 | FileCheck %s

@.str = private unnamed_addr addrspace(2) constant [6 x i8] c"%s:%d\00", align 1

define amdgpu_kernel void @test_kernel(i32 %n) {
entry:
  %str = alloca [9 x i8], align 1
  %arraydecay = getelementptr inbounds [9 x i8], [9 x i8]* %str, i32 0, i32 0
  %call1 = call i32 (i8 addrspace(2)*, ...) @printf(i8 addrspace(2)* getelementptr inbounds ([6 x i8], [6 x i8] addrspace(2)* @.str, i32 0, i32 0), i8* %arraydecay, i32 %n)
  %call2 = call <2 x i64> (i8*, i32, i64, i64, i64, i64, i64, i64, i64, i64) @__ockl_hostcall_internal(i8* undef, i32 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i64 9)
  ret void
}

declare i32 @printf(i8 addrspace(2)*, ...)

declare <2 x i64> @__ockl_hostcall_internal(i8*, i32, i64, i64, i64, i64, i64, i64, i64, i64)

; CHECK: error: Cannot use both printf and hostcall in the same module
