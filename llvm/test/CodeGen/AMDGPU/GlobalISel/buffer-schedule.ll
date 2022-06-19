; RUN: llc -global-isel -mtriple=amdgcn-amd-amdpal -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN %s

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

declare i32 @llvm.amdgcn.raw.buffer.load.i32(<4 x i32>, i32, i32, i32) #2

declare void @llvm.amdgcn.raw.buffer.store.i32(i32, <4 x i32>, i32, i32, i32) #3

attributes #2 = { nounwind readonly }
attributes #3 = { nounwind writeonly }
