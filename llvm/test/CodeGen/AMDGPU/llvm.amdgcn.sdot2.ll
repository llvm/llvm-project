; RUN: llc -march=amdgcn -mcpu=gfx906 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=GCN,GFX906
; RUN: llc -march=amdgcn -mcpu=gfx908 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=GCN,GFX908
; RUN: llc -march=amdgcn -mcpu=gfx1011 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=GCN,GFX10
; RUN: llc -march=amdgcn -mcpu=gfx1012 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=GCN,GFX10

declare i32 @llvm.amdgcn.sdot2(<2 x i16> %a, <2 x i16> %b, i32 %c, i1 %clamp)

; GCN-LABEL: {{^}}test_llvm_amdgcn_sdot2_clamp
; GFX906: v_dot2_i32_i16 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} clamp{{$}}
; GFX908: v_dot2_i32_i16 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} clamp{{$}}
; GFX10:  v_dot2_i32_i16 v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}} clamp{{$}}
define amdgpu_kernel void @test_llvm_amdgcn_sdot2_clamp(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b,
    ptr addrspace(1) %c) {
entry:
  %a.val = load <2 x i16>, ptr addrspace(1) %a
  %b.val = load <2 x i16>, ptr addrspace(1) %b
  %c.val = load i32, ptr addrspace(1) %c
  %r.val = call i32 @llvm.amdgcn.sdot2(<2 x i16> %a.val, <2 x i16> %b.val, i32 %c.val, i1 1)
  store i32 %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}test_llvm_amdgcn_sdot2_no_clamp
; GFX906: v_dot2_i32_i16 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}{{$}}
; GFX908: v_dot2c_i32_i16_e32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}{{$}}
; GFX10:  v_dot2_i32_i16 v{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}{{$}}
define amdgpu_kernel void @test_llvm_amdgcn_sdot2_no_clamp(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b,
    ptr addrspace(1) %c) {
entry:
  %a.val = load <2 x i16>, ptr addrspace(1) %a
  %b.val = load <2 x i16>, ptr addrspace(1) %b
  %c.val = load i32, ptr addrspace(1) %c
  %r.val = call i32 @llvm.amdgcn.sdot2(<2 x i16> %a.val, <2 x i16> %b.val, i32 %c.val, i1 0)
  store i32 %r.val, ptr addrspace(1) %r
  ret void
}
