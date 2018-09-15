; RUN: llc -march=amdgcn -mcpu=gfx906 -verify-machineinstrs < %s | FileCheck %s --check-prefix=GCN --check-prefix=GFX906

declare i32 @llvm.amdgcn.sdot8(i32 %a, i32 %b, i32 %c, i1 %clamp)

; GCN-LABEL: {{^}}test_llvm_amdgcn_sdot8_clamp
; GFX906: v_dot8_i32_i4 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} clamp{{$}}
define amdgpu_kernel void @test_llvm_amdgcn_sdot8_clamp(
    i32 addrspace(1)* %r,
    <8 x i4> addrspace(1)* %a,
    <8 x i4> addrspace(1)* %b,
    i32 addrspace(1)* %c) {
entry:
  %a.val = load <8 x i4>, <8 x i4> addrspace(1)* %a
  %b.val = load <8 x i4>, <8 x i4> addrspace(1)* %b
  %a.val.cast = bitcast <8 x i4> %a.val to i32
  %b.val.cast = bitcast <8 x i4> %b.val to i32
  %c.val = load i32, i32 addrspace(1)* %c
  %r.val = call i32 @llvm.amdgcn.sdot8(i32 %a.val.cast, i32 %b.val.cast, i32 %c.val, i1 1)
  store i32 %r.val, i32 addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}test_llvm_amdgcn_sdot8_no_clamp
; GFX906: v_dot8_i32_i4 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}{{$}}
define amdgpu_kernel void @test_llvm_amdgcn_sdot8_no_clamp(
    i32 addrspace(1)* %r,
    <8 x i4> addrspace(1)* %a,
    <8 x i4> addrspace(1)* %b,
    i32 addrspace(1)* %c) {
entry:
  %a.val = load <8 x i4>, <8 x i4> addrspace(1)* %a
  %b.val = load <8 x i4>, <8 x i4> addrspace(1)* %b
  %a.val.cast = bitcast <8 x i4> %a.val to i32
  %b.val.cast = bitcast <8 x i4> %b.val to i32
  %c.val = load i32, i32 addrspace(1)* %c
  %r.val = call i32 @llvm.amdgcn.sdot8(i32 %a.val.cast, i32 %b.val.cast, i32 %c.val, i1 0)
  store i32 %r.val, i32 addrspace(1)* %r
  ret void
}
