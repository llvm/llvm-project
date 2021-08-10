; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx803 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN %s

; The buffer_loads and buffer_stores all access the same location. Check they do
; not get reordered by the scheduler.

; GCN-LABEL: {{^}}_amdgpu_cs_main:
; GCN: buffer_load_dword
; GCN: buffer_store_dword
; GCN: buffer_load_dword
; GCN: buffer_store_dword
; GCN: buffer_load_dword
; GCN: buffer_store_dword
; GCN: buffer_load_dword
; GCN: buffer_store_dword

; Function Attrs: nounwind
define amdgpu_cs void @_amdgpu_cs_main(<3 x i32> inreg %arg3, <3 x i32> %arg5) {
.entry:
  %tmp9 = add <3 x i32> %arg3, %arg5
  %tmp10 = extractelement <3 x i32> %tmp9, i32 0
  %tmp11 = shl i32 %tmp10, 2
  %tmp12 = inttoptr i64 undef to <4 x i32> addrspace(4)*
  %tmp13 = load <4 x i32>, <4 x i32> addrspace(4)* %tmp12, align 16
  %tmp14 = call float @llvm.amdgcn.buffer.load.f32(<4 x i32> %tmp13, i32 0, i32 %tmp11, i1 false, i1 false) #0
  %tmp17 = load <4 x i32>, <4 x i32> addrspace(4)* %tmp12, align 16
  call void @llvm.amdgcn.buffer.store.f32(float %tmp14, <4 x i32> %tmp17, i32 0, i32 %tmp11, i1 false, i1 false) #0
  %tmp20 = load <4 x i32>, <4 x i32> addrspace(4)* %tmp12, align 16
  %tmp21 = call float @llvm.amdgcn.buffer.load.f32(<4 x i32> %tmp20, i32 0, i32 %tmp11, i1 false, i1 false) #0
  %tmp22 = fadd reassoc nnan arcp contract float %tmp21, 1.000000e+00
  call void @llvm.amdgcn.buffer.store.f32(float %tmp22, <4 x i32> %tmp20, i32 0, i32 %tmp11, i1 false, i1 false) #0
  %tmp25 = load <4 x i32>, <4 x i32> addrspace(4)* %tmp12, align 16
  %tmp26 = call float @llvm.amdgcn.buffer.load.f32(<4 x i32> %tmp25, i32 0, i32 %tmp11, i1 false, i1 false) #0
  %tmp27 = fadd reassoc nnan arcp contract float %tmp26, 1.000000e+00
  call void @llvm.amdgcn.buffer.store.f32(float %tmp27, <4 x i32> %tmp25, i32 0, i32 %tmp11, i1 false, i1 false) #0
  %tmp30 = load <4 x i32>, <4 x i32> addrspace(4)* %tmp12, align 16
  %tmp31 = call float @llvm.amdgcn.buffer.load.f32(<4 x i32> %tmp30, i32 0, i32 %tmp11, i1 false, i1 false) #0
  %tmp32 = fadd reassoc nnan arcp contract float %tmp31, 1.000000e+00
  call void @llvm.amdgcn.buffer.store.f32(float %tmp32, <4 x i32> %tmp30, i32 0, i32 %tmp11, i1 false, i1 false) #0
  %tmp35 = load <4 x i32>, <4 x i32> addrspace(4)* %tmp12, align 16
  %tmp36 = call float @llvm.amdgcn.buffer.load.f32(<4 x i32> %tmp35, i32 0, i32 %tmp11, i1 false, i1 false) #0
  %tmp37 = fadd reassoc nnan arcp contract float %tmp36, 1.000000e+00
  call void @llvm.amdgcn.buffer.store.f32(float %tmp37, <4 x i32> %tmp35, i32 0, i32 %tmp11, i1 false, i1 false) #0
  ret void
}

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

declare float @llvm.amdgcn.buffer.load.f32(<4 x i32>, i32, i32, i1, i1) #2

declare void @llvm.amdgcn.buffer.store.f32(float, <4 x i32>, i32, i32, i1, i1) #3

declare i32 @llvm.amdgcn.raw.buffer.load.i32(<4 x i32>, i32, i32, i32) #2

declare void @llvm.amdgcn.raw.buffer.store.i32(i32, <4 x i32>, i32, i32, i32) #3

attributes #2 = { nounwind readonly }
attributes #3 = { nounwind writeonly }
