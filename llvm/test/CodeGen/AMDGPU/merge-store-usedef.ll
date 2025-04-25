; RUN: llc -mtriple=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck %s

; CHECK-LABEL: {{^}}test1:
; CHECK: ds_write_b32
; CHECK: ds_read_b32
; CHECK: ds_write_b32
define amdgpu_vs void @test1(i32 %v) #0 {
  %p1 = getelementptr i32, ptr addrspace(3) null, i32 1

  store i32 %v, ptr addrspace(3) null

  call void @llvm.amdgcn.raw.ptr.tbuffer.store.i32(i32 %v, ptr addrspace(8) undef, i32 0, i32 0, i32 68, i32 1)

  %w = load i32, ptr addrspace(3) null
  store i32 %w, ptr addrspace(3) %p1
  ret void
}

declare void @llvm.amdgcn.raw.ptr.tbuffer.store.i32(i32, ptr addrspace(8), i32, i32, i32 immarg, i32 immarg) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind willreturn writeonly }
