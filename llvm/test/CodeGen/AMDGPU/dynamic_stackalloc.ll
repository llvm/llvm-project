; RUN: not llc -mtriple=amdgcn-- -mcpu=tahiti -mattr=+promote-alloca -verify-machineinstrs < %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple=amdgcn-- -mcpu=tahiti -mattr=-promote-alloca -verify-machineinstrs < %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple=r600-- -mcpu=cypress < %s 2>&1 | FileCheck %s
target datalayout = "A5"

; CHECK: in function test_dynamic_stackalloc{{.*}}: unsupported dynamic alloca

define amdgpu_kernel void @test_dynamic_stackalloc(ptr addrspace(1) %out, i32 %n) {
  %alloca = alloca i32, i32 %n, addrspace(5)
  store volatile i32 0, ptr addrspace(5) %alloca
  ret void
}
