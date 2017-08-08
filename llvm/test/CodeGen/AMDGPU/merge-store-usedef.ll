; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck %s

; CHECK-LABEL: {{^}}test1:
; CHECK: ds_write_b32
; CHECK: ds_read_b32
; CHECK: ds_write_b32
define amdgpu_vs void @test1(i32 %v) #0 {
  %p0 = getelementptr i32, i32 addrspace(3)* null, i32 0
  %p1 = getelementptr i32, i32 addrspace(3)* null, i32 1

  store i32 %v, i32 addrspace(3)* %p0

  call void @llvm.amdgcn.tbuffer.store.i32(i32 %v, <4 x i32> undef, i32 0, i32 0, i32 0, i32 0, i32 4, i32 4, i1 1, i1 0)

  %w = load i32, i32 addrspace(3)* %p0
  store i32 %w, i32 addrspace(3)* %p1
  ret void
}

declare void @llvm.amdgcn.tbuffer.store.i32(i32, <4 x i32>, i32, i32, i32, i32, i32, i32, i1, i1) #0

attributes #0 = { nounwind }
