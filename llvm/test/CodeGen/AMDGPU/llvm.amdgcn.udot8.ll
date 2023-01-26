; RUN: llc -march=amdgcn -mcpu=gfx906 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=GCN,GFX9
; RUN: llc -march=amdgcn -mcpu=gfx940 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=GCN,GFX9
; RUN: llc -march=amdgcn -mcpu=gfx1011 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=GCN,GFX10
; RUN: llc -march=amdgcn -mcpu=gfx1012 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=GCN,GFX10
; RUN: llc -march=amdgcn -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=GCN,GFX10

declare i32 @llvm.amdgcn.udot8(i32 %a, i32 %b, i32 %c, i1 %clamp)

; GCN-LABEL: {{^}}test_llvm_amdgcn_udot8_clamp
; GFX9:   v_dot8_u32_u4 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} clamp{{$}}
; GFX10:  v_dot8_u32_u4 v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}} clamp{{$}}
define amdgpu_kernel void @test_llvm_amdgcn_udot8_clamp(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b,
    ptr addrspace(1) %c) {
entry:
  %a.val = load <8 x i4>, ptr addrspace(1) %a
  %b.val = load <8 x i4>, ptr addrspace(1) %b
  %a.val.cast = bitcast <8 x i4> %a.val to i32
  %b.val.cast = bitcast <8 x i4> %b.val to i32
  %c.val = load i32, ptr addrspace(1) %c
  %r.val = call i32 @llvm.amdgcn.udot8(i32 %a.val.cast, i32 %b.val.cast, i32 %c.val, i1 1)
  store i32 %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}test_llvm_amdgcn_udot8_no_clamp
; GFX9:   v_dot8_u32_u4 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}{{$}}
; GFX10:  v_dot8_u32_u4 v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}{{$}}
define amdgpu_kernel void @test_llvm_amdgcn_udot8_no_clamp(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b,
    ptr addrspace(1) %c) {
entry:
  %a.val = load <8 x i4>, ptr addrspace(1) %a
  %b.val = load <8 x i4>, ptr addrspace(1) %b
  %a.val.cast = bitcast <8 x i4> %a.val to i32
  %b.val.cast = bitcast <8 x i4> %b.val to i32
  %c.val = load i32, ptr addrspace(1) %c
  %r.val = call i32 @llvm.amdgcn.udot8(i32 %a.val.cast, i32 %b.val.cast, i32 %c.val, i1 0)
  store i32 %r.val, ptr addrspace(1) %r
  ret void
}
