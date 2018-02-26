; RUN: not llc -march=amdgcn -mtriple=amdgcn---amdgiz -mcpu=tahiti -mattr=+promote-alloca -verify-machineinstrs < %s 2>&1 | FileCheck %s
; RUN: not llc -march=amdgcn -mtriple=amdgcn---amdgiz -mcpu=tahiti -mattr=-promote-alloca -verify-machineinstrs < %s 2>&1 | FileCheck %s
; RUN: not llc -march=r600 -mtriple=r600---amdgiz -mcpu=cypress < %s 2>&1 | FileCheck %s
target datalayout = "A5"

; CHECK: in function test_dynamic_stackalloc{{.*}}: unsupported dynamic alloca

define amdgpu_kernel void @test_dynamic_stackalloc(i32 addrspace(1)* %out, i32 %n) {
  %alloca = alloca i32, i32 %n, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca
  ret void
}
