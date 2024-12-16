; RUN: not llc -mtriple=amdgcn-- -mcpu=tahiti -mattr=+promote-alloca -verify-machineinstrs < %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple=amdgcn-- -mcpu=tahiti -mattr=-promote-alloca -verify-machineinstrs < %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple=r600-- -mcpu=cypress < %s 2>&1 | FileCheck %s
target datalayout = "A5"

; CHECK: in function test_dynamic_stackalloc{{.*}}: unsupported dynamic alloca

define amdgpu_kernel void @test_dynamic_stackalloc(i32 %n) {
  %alloca = alloca i32, i32 %n, addrspace(5)
  store volatile i32 123, ptr addrspace(5) %alloca
  ret void
}

; CHECK: in function test_dynamic_stackalloc{{.*}}: unsupported dynamic alloca

define amdgpu_kernel void @test_dynamic_stackalloc_multiple_allocas(i32 %n) {
  %alloca1 = alloca i32, i32 8, addrspace(5)
  %alloca2 = alloca i32, i32 %n, addrspace(5)
  %alloca3 = alloca i32, i32 10, addrspace(5)
  store volatile i32 1, ptr addrspace(5) %alloca1
  store volatile i32 2, ptr addrspace(5) %alloca2
  store volatile i32 3, ptr addrspace(5) %alloca3
  ret void
}

; CHECK: in function test_dynamic_stackalloc{{.*}}: unsupported dynamic alloca

define amdgpu_kernel void @test_dynamic_stackalloc_custom_alignment(i32 %n) {
  %alloca = alloca i32, i32 %n, align 128, addrspace(5)
  store volatile i32 1, ptr addrspace(5) %alloca
  ret void
}

; CHECK: in function test_dynamic_stackalloc{{.*}}: unsupported dynamic alloca

define amdgpu_kernel void @test_dynamic_stackalloc_non_entry_block(i32 %n) {
  entry:
    %cond = icmp eq i32 %n, 0
    br i1 %cond, label %bb.0, label %bb.1

  bb.0:
    %alloca = alloca i32, i32 %n, align 64, addrspace(5)
    %gep1 = getelementptr i32, ptr addrspace(5) %alloca, i32 1
    store volatile i32 0, ptr addrspace(5) %alloca
    store volatile i32 1, ptr addrspace(5) %gep1
    br label %bb.1

  bb.1:
    ret void
}
