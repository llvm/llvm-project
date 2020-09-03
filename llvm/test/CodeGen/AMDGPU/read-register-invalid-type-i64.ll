; RUN: not --crash llc -march=amdgcn -verify-machineinstrs < %s 2>&1 | FileCheck %s

; CHECK: invalid type for register "m0".

declare i64 @llvm.read_register.i64(metadata) #0

define amdgpu_kernel void @test_invalid_read_m0(i64 addrspace(1)* %out) #0 {
  %exec = call i64 @llvm.read_register.i64(metadata !0)
  store i64 %exec, i64 addrspace(1)* %out
  ret void
}

!0 = !{!"m0"}
