; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1300 -verify-machineinstrs -o - %s | FileCheck -check-prefix=VGPR %s

define amdgpu_kernel void @test_bpermute_sv(ptr addrspace(1) %out, i32 inreg %src0, i32 %src1) "amdgpu-wavegroup-enable" !reqd_work_group_size !{i32 32, i32 12, i32 1} {
  %res = call i32 @llvm.amdgcn.bpermute.b32(i32 %src0, i32 %src1)
  store i32 %res, ptr addrspace(1) %out
  ret void
}

; VGPR: ; NumVgprs: 6
; VGPR: ; NumVGPRsForWavesPerEU: 6
