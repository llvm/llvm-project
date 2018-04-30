; RUN: llc -march=amdgcn -mcpu=gfx906 -verify-machineinstrs < %s | FileCheck %s --check-prefix=GCN --check-prefix=GFX906

declare i32 @llvm.amdgcn.udot2(<2 x i16> %a, <2 x i16> %b, i32 %c)

; GCN-LABEL: {{^}}test_llvm_amdgcn_udot2
; GFX906: v_dot2_u32_u16
define amdgpu_kernel void @test_llvm_amdgcn_udot2(
    i32 addrspace(1)* %r,
    <2 x i16> addrspace(1)* %a,
    <2 x i16> addrspace(1)* %b,
    i32 addrspace(1)* %c) {
entry:
  %a.val = load <2 x i16>, <2 x i16> addrspace(1)* %a
  %b.val = load <2 x i16>, <2 x i16> addrspace(1)* %b
  %c.val = load i32, i32 addrspace(1)* %c
  %r.val = call i32 @llvm.amdgcn.udot2(<2 x i16> %a.val, <2 x i16> %b.val, i32 %c.val)
  store i32 %r.val, i32 addrspace(1)* %r
  ret void
}
