; RUN: not llc -mtriple=amdgcn-- -mcpu=tahiti -mattr=+promote-alloca -verify-machineinstrs < %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple=amdgcn-- -mcpu=tahiti -mattr=-promote-alloca -verify-machineinstrs < %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple=r600-- -mcpu=cypress < %s 2>&1 | FileCheck %s
target datalayout = "A5"

; CHECK: in function test_dynamic_stackalloc{{.*}}: unsupported dynamic alloca

define amdgpu_kernel void @test_dynamic_stackalloc_kernel_uniform(i32 %n) {
  %alloca = alloca i32, i32 %n, addrspace(5)
  store volatile i32 123, ptr addrspace(5) %alloca
  ret void
}

; CHECK: in function test_dynamic_stackalloc{{.*}}: unsupported dynamic alloca

define amdgpu_kernel void @test_dynamic_stackalloc_kernel_uniform_over_aligned(i32 %n) {
  %alloca = alloca i32, i32 %n, align 128, addrspace(5)
  store volatile i32 10, ptr addrspace(5) %alloca
  ret void
}

; CHECK: in function test_dynamic_stackalloc{{.*}}: unsupported dynamic alloca

define amdgpu_kernel void @test_dynamic_stackalloc_kernel_uniform_under_aligned(i32 %n) {
  %alloca = alloca i32, i32 %n, align 2, addrspace(5)
  store volatile i32 22, ptr addrspace(5) %alloca
  ret void
}

; CHECK: in function test_dynamic_stackalloc{{.*}}: unsupported dynamic alloca

define amdgpu_kernel void @test_dynamic_stackalloc_kernel_divergent() {
  %idx = call i32 @llvm.amdgcn.workitem.id.x()
  %alloca = alloca float, i32 %idx, addrspace(5)
  store volatile i32 123, ptr addrspace(5) %alloca
  ret void
}

; CHECK: in function test_dynamic_stackalloc{{.*}}: unsupported dynamic alloca

define amdgpu_kernel void @test_dynamic_stackalloc_kernel_divergent_over_aligned() {
  %idx = call i32 @llvm.amdgcn.workitem.id.x()
  %alloca = alloca i32, i32 %idx, align 128, addrspace(5)
  store volatile i32 444, ptr addrspace(5) %alloca
  ret void
}

; CHECK: in function test_dynamic_stackalloc{{.*}}: unsupported dynamic alloca

define amdgpu_kernel void @test_dynamic_stackalloc_kernel_divergent_under_aligned() {
  %idx = call i32 @llvm.amdgcn.workitem.id.x()
  %alloca = alloca i128, i32 %idx, align 2, addrspace(5)
  store volatile i32 666, ptr addrspace(5) %alloca
  ret void
}

; CHECK: in function test_dynamic_stackalloc{{.*}}: unsupported dynamic alloca
; CHECK: in function test_dynamic_stackalloc{{.*}}: unsupported dynamic alloca
; CHECK: in function test_dynamic_stackalloc{{.*}}: unsupported dynamic alloca

define amdgpu_kernel void @test_dynamic_stackalloc_kernel_multiple_allocas(i32 %n, i32 %m) {
entry:
  %cond = icmp eq i32 %n, 0
  %alloca1 = alloca i32, i32 8, addrspace(5)
  %alloca2 = alloca i17, i32 %n, addrspace(5)
  br i1 %cond, label %bb.0, label %bb.1
bb.0:
  %idx = call i32 @llvm.amdgcn.workitem.id.x()
  %alloca3 = alloca i32, i32 %m, align 64, addrspace(5)
  %alloca4 = alloca i32, i32 %idx, align 4, addrspace(5)
  store volatile i32 3, ptr addrspace(5) %alloca3
  store volatile i32 4, ptr addrspace(5) %alloca4
  br label %bb.1
bb.1:
  store volatile i32 1, ptr addrspace(5) %alloca1
  store volatile i32 2, ptr addrspace(5) %alloca2
  ret void
}

; CHECK: in function test_dynamic_stackalloc{{.*}}: unsupported dynamic alloca
; CHECK: in function test_dynamic_stackalloc{{.*}}: unsupported dynamic alloca

define amdgpu_kernel void @test_dynamic_stackalloc_kernel_control_flow(i32 %n, i32 %m) {
entry:
  %cond = icmp eq i32 %n, 0
  br i1 %cond, label %bb.0, label %bb.1
bb.0:
  %alloca2 = alloca i32, i32 %m, align 64, addrspace(5)
  store volatile i32 2, ptr addrspace(5) %alloca2
  br label %bb.2
bb.1:
  %idx = call i32 @llvm.amdgcn.workitem.id.x()
  %alloca1 = alloca i32, i32 %idx, align 4, addrspace(5)
  store volatile i32 1, ptr addrspace(5) %alloca1
  br label %bb.2
bb.2:
  ret void
}

; CHECK: in function test_dynamic_stackalloc{{.*}}: unsupported dynamic alloca

define void @test_dynamic_stackalloc_device_uniform(i32 %n) {
  %alloca = alloca i32, i32 %n, addrspace(5)
  store volatile i32 123, ptr addrspace(5) %alloca
  ret void
}

; CHECK: in function test_dynamic_stackalloc{{.*}}: unsupported dynamic alloca

define void @test_dynamic_stackalloc_device_uniform_over_aligned(i32 %n) {
  %alloca = alloca i32, i32 %n, align 128, addrspace(5)
  store volatile i32 10, ptr addrspace(5) %alloca
  ret void
}

; CHECK: in function test_dynamic_stackalloc{{.*}}: unsupported dynamic alloca

define void @test_dynamic_stackalloc_device_uniform_under_aligned(i32 %n) {
  %alloca = alloca i32, i32 %n, align 2, addrspace(5)
  store volatile i32 22, ptr addrspace(5) %alloca
  ret void
}

; CHECK: in function test_dynamic_stackalloc{{.*}}: unsupported dynamic alloca

define void @test_dynamic_stackalloc_device_divergent() {
  %idx = call i32 @llvm.amdgcn.workitem.id.x()
  %alloca = alloca i32, i32 %idx, addrspace(5)
  store volatile i32 123, ptr addrspace(5) %alloca
  ret void
}

; CHECK: in function test_dynamic_stackalloc{{.*}}: unsupported dynamic alloca

define void @test_dynamic_stackalloc_device_divergent_over_aligned() {
  %idx = call i32 @llvm.amdgcn.workitem.id.x()
  %alloca = alloca i32, i32 %idx, align 128, addrspace(5)
  store volatile i32 444, ptr addrspace(5) %alloca
  ret void
}

; CHECK: in function test_dynamic_stackalloc{{.*}}: unsupported dynamic alloca

define void @test_dynamic_stackalloc_device_divergent_under_aligned() {
  %idx = call i32 @llvm.amdgcn.workitem.id.x()
  %alloca = alloca i32, i32 %idx, align 2, addrspace(5)
  store volatile i32 666, ptr addrspace(5) %alloca
  ret void
}

; CHECK: in function test_dynamic_stackalloc{{.*}}: unsupported dynamic alloca
; CHECK: in function test_dynamic_stackalloc{{.*}}: unsupported dynamic alloca
; CHECK: in function test_dynamic_stackalloc{{.*}}: unsupported dynamic alloca

define void @test_dynamic_stackalloc_device_multiple_allocas(i32 %n, i32 %m) {
entry:
  %cond = icmp eq i32 %n, 0
  %alloca1 = alloca i32, i32 8, addrspace(5)
  %alloca2 = alloca i32, i32 %n, addrspace(5)
  br i1 %cond, label %bb.0, label %bb.1
bb.0:
  %idx = call i32 @llvm.amdgcn.workitem.id.x()
  %alloca3 = alloca i32, i32 %m, align 64, addrspace(5)
  %alloca4 = alloca i32, i32 %idx, align 4, addrspace(5)
  store volatile i32 3, ptr addrspace(5) %alloca3
  store volatile i32 4, ptr addrspace(5) %alloca4
  br label %bb.1
bb.1:
  store volatile i32 1, ptr addrspace(5) %alloca1
  store volatile i32 2, ptr addrspace(5) %alloca2
  ret void
}

; CHECK: in function test_dynamic_stackalloc{{.*}}: unsupported dynamic alloca
; CHECK: in function test_dynamic_stackalloc{{.*}}: unsupported dynamic alloca

define void @test_dynamic_stackalloc_device_control_flow(i32 %n, i32 %m) {
entry:
  %cond = icmp eq i32 %n, 0
  br i1 %cond, label %bb.0, label %bb.1
bb.0:
  %idx = call i32 @llvm.amdgcn.workitem.id.x()
  %alloca1 = alloca i32, i32 %idx, align 4, addrspace(5)
  store volatile i32 1, ptr addrspace(5) %alloca1
  br label %bb.2
bb.1:
  %alloca2 = alloca i32, i32 %m, align 64, addrspace(5)
  store volatile i32 2, ptr addrspace(5) %alloca2
  br label %bb.2
bb.2:
  ret void
}
