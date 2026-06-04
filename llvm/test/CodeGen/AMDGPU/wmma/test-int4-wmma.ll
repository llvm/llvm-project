; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1200 -verify-machineinstrs < %s | FileCheck %s
; REQUIRES: amdgpu-registered-target
;
; Test: WMMA + SWMMAC INT4/INT8 instruction generation for gfx1200
; Verifies our WMMA256bInsts + Wave32 patches produce correct machine code.

; CHECK-LABEL: test_wmma_i32_16x16x16_iu4:
; CHECK: v_wmma_i32_16x16x16_iu4
define amdgpu_kernel void @test_wmma_i32_16x16x16_iu4(ptr addrspace(1) %C, ptr addrspace(1) %A, ptr addrspace(1) %B) #0 {
  %a.val = load i32, ptr addrspace(1) %A
  %b.val = load i32, ptr addrspace(1) %B
  %c = load <8 x i32>, ptr addrspace(1) %C
  %res = call <8 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu4.v8i32.i32(i1 false, i32 %a.val, i1 false, i32 %b.val, <8 x i32> %c, i1 false)
  store <8 x i32> %res, ptr addrspace(1) %C
  ret void
}

; CHECK-LABEL: test_swmmac_i32_16x16x32_iu8:
; CHECK: v_swmmac_i32_16x16x32_iu8
define amdgpu_kernel void @test_swmmac_i32_16x16x32_iu8(ptr addrspace(1) %C, ptr addrspace(1) %A, ptr addrspace(1) %B) #0 {
  %a = load <2 x i32>, ptr addrspace(1) %A
  %b = load <4 x i32>, ptr addrspace(1) %B
  %c = load <8 x i32>, ptr addrspace(1) %C
  %res = call <8 x i32> @llvm.amdgcn.swmmac.i32.16x16x32.iu8.v8i32.v2i32.v4i32.i32(i1 false, <2 x i32> %a, i1 false, <4 x i32> %b, <8 x i32> %c, i32 0, i1 false)
  store <8 x i32> %res, ptr addrspace(1) %C
  ret void
}

; CHECK-LABEL: test_swmmac_i32_16x16x64_iu4:
; CHECK: v_swmmac_i32_16x16x64_iu4
define amdgpu_kernel void @test_swmmac_i32_16x16x64_iu4(ptr addrspace(1) %C, ptr addrspace(1) %A, ptr addrspace(1) %B) #0 {
  %a = load <2 x i32>, ptr addrspace(1) %A
  %b = load <4 x i32>, ptr addrspace(1) %B
  %c = load <8 x i32>, ptr addrspace(1) %C
  %res = call <8 x i32> @llvm.amdgcn.swmmac.i32.16x16x64.iu4.v8i32.v2i32.v4i32.i32(i1 false, <2 x i32> %a, i1 false, <4 x i32> %b, <8 x i32> %c, i32 0, i1 false)
  store <8 x i32> %res, ptr addrspace(1) %C
  ret void
}

declare <8 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu4.v8i32.i32(i1 immarg, i32, i1 immarg, i32, <8 x i32>, i1 immarg)
declare <8 x i32> @llvm.amdgcn.swmmac.i32.16x16x32.iu8.v8i32.v2i32.v4i32.i32(i1 immarg, <2 x i32>, i1 immarg, <4 x i32>, <8 x i32>, i32, i1 immarg)
declare <8 x i32> @llvm.amdgcn.swmmac.i32.16x16x64.iu4.v8i32.v2i32.v4i32.i32(i1 immarg, <2 x i32>, i1 immarg, <4 x i32>, <8 x i32>, i32, i1 immarg)

attributes #0 = { "target-features"="+wavefrontsize32" }
