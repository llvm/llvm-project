; RUN: llc -march=amdgcn -mcpu=gfx906 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=GCN,GFX906
; RUN: llc -march=amdgcn -mcpu=gfx1011 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=GCN,GFX10
; RUN: llc -march=amdgcn -mcpu=gfx1012 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=GCN,GFX10
; RUN: llc -march=amdgcn -mcpu=gfx1030 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=GCN,GFX10
; RUN: llc -march=amdgcn -mcpu=gfx1031 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=GCN,GFX10

declare i32 @llvm.amdgcn.sdot4(i32 %a, i32 %b, i32 %c, i1 %clamp)

; GCN-LABEL: {{^}}test_llvm_amdgcn_sdot4_clamp
; GFX906: v_dot4_i32_i8 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} clamp{{$}}
; GFX10:  v_dot4_i32_i8 v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}} clamp{{$}}
define amdgpu_kernel void @test_llvm_amdgcn_sdot4_clamp(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b,
    ptr addrspace(1) %c) {
entry:
  %a.val = load <4 x i8>, ptr addrspace(1) %a
  %b.val = load <4 x i8>, ptr addrspace(1) %b
  %a.val.cast = bitcast <4 x i8> %a.val to i32
  %b.val.cast = bitcast <4 x i8> %b.val to i32
  %c.val = load i32, ptr addrspace(1) %c
  %r.val = call i32 @llvm.amdgcn.sdot4(i32 %a.val.cast, i32 %b.val.cast, i32 %c.val, i1 1)
  store i32 %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}test_llvm_amdgcn_sdot4_no_clamp
; GFX906: v_dot4_i32_i8 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}{{$}}
; GFX10:  v_dot4c_i32_i8_e32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}{{$}}
define amdgpu_kernel void @test_llvm_amdgcn_sdot4_no_clamp(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b,
    ptr addrspace(1) %c) {
entry:
  %a.val = load <4 x i8>, ptr addrspace(1) %a
  %b.val = load <4 x i8>, ptr addrspace(1) %b
  %a.val.cast = bitcast <4 x i8> %a.val to i32
  %b.val.cast = bitcast <4 x i8> %b.val to i32
  %c.val = load i32, ptr addrspace(1) %c
  %r.val = call i32 @llvm.amdgcn.sdot4(i32 %a.val.cast, i32 %b.val.cast, i32 %c.val, i1 0)
  store i32 %r.val, ptr addrspace(1) %r
  ret void
}
