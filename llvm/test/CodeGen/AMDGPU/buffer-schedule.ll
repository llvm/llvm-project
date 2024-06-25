; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx803 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN %s

; The buffer_loads and buffer_stores all access the same location. Check they do
; not get reordered by the scheduler.

; GCN-LABEL: {{^}}test1:
; GCN: buffer_store_dword
; GCN: buffer_load_dword
; GCN: buffer_store_dword
define amdgpu_cs void @test1(<4 x i32> inreg %buf, i32 %off) {
.entry:
  call void @llvm.amdgcn.raw.buffer.store.i32(i32 0, <4 x i32> %buf, i32 8, i32 0, i32 0)
  %val = call i32 @llvm.amdgcn.raw.buffer.load.i32(<4 x i32> %buf, i32 %off, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.store.i32(i32 %val, <4 x i32> %buf, i32 0, i32 0, i32 0)
  ret void
}

; GCN-LABEL: {{^}}test1_ptrs:
; GCN: buffer_store_dword
; GCN: buffer_load_dword
; GCN: buffer_store_dword
define amdgpu_cs void @test1_ptrs(ptr addrspace(8) inreg %buf, i32 %off) {
.entry:
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 0, ptr addrspace(8) %buf, i32 8, i32 0, i32 0)
  %val = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %buf, i32 %off, i32 0, i32 0)
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %val, ptr addrspace(8) %buf, i32 0, i32 0, i32 0)
  ret void
}

;; In the future, the stores should be reorderable because they'd be known to be
;; at distinct offsets.
; GCN-LABEL: {{^}}test1_ptrs_reorderable:
; GCN: buffer_store_dword
; GCN: buffer_load_dword
; GCN: buffer_store_dword
define amdgpu_cs void @test1_ptrs_reorderable(ptr addrspace(8) inreg %buf, i32 %off) {
.entry:
  %shifted.off = shl i32 %off, 4
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 0, ptr addrspace(8) %buf, i32 8, i32 0, i32 0)
  %val = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %buf, i32 %shifted.off, i32 0, i32 0)
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %val, ptr addrspace(8) %buf, i32 0, i32 0, i32 0)
  ret void
}


declare i32 @llvm.amdgcn.raw.buffer.load.i32(<4 x i32>, i32, i32, i32) #2

declare void @llvm.amdgcn.raw.buffer.store.i32(i32, <4 x i32>, i32, i32, i32) #3

declare i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) nocapture, i32, i32, i32) #4

declare void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32, ptr addrspace(8) nocapture, i32, i32, i32) #5

attributes #2 = { nounwind readonly }
attributes #3 = { nounwind writeonly }
attributes #4 = { nounwind memory(argmem: read) }
attributes #5 = { nounwind memory(argmem: write) }
