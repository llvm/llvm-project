; RUN: split-file %s %t
; RUN: llc -mtriple=amdgcn -mcpu=gfx908 < %t/valid.ll | FileCheck --check-prefix=GCN %s
; RUN: not --crash llc -mtriple=amdgcn -mcpu=gfx908 -verify-machineinstrs < %t/vmem.ll 2>&1 | FileCheck --check-prefix=CHECK-ERR %s
; RUN: not --crash llc -mtriple=amdgcn -mcpu=gfx908 -verify-machineinstrs < %t/vmem-read.ll 2>&1 | FileCheck --check-prefix=CHECK-ERR %s
; RUN: not --crash llc -mtriple=amdgcn -mcpu=gfx908 -verify-machineinstrs < %t/vmem-write.ll 2>&1 | FileCheck --check-prefix=CHECK-ERR %s
; RUN: not --crash llc -mtriple=amdgcn -mcpu=gfx908 -verify-machineinstrs < %t/ds.ll 2>&1 | FileCheck --check-prefix=CHECK-ERR %s
; RUN: not --crash llc -mtriple=amdgcn -mcpu=gfx908 -verify-machineinstrs < %t/ds-read.ll 2>&1 | FileCheck --check-prefix=CHECK-ERR %s
; RUN: not --crash llc -mtriple=amdgcn -mcpu=gfx908 -verify-machineinstrs < %t/ds-write.ll 2>&1 | FileCheck --check-prefix=CHECK-ERR %s
; RUN: not --crash llc -mtriple=amdgcn -mcpu=gfx908 -verify-machineinstrs < %t/ldsdma.ll 2>&1 | FileCheck --check-prefix=CHECK-ERR %s

; CHECK-ERR: S_SETPRIO mask contains invalid memory operation bits

; Demonstrate that the maks bits that are compatible with s_set_prio
; allow corresponding instruction types to be scheduled across the s_setprio
; instruction and that incompatible mask bits are rejected by the
; verifier.

;--- valid.ll

declare void @llvm.amdgcn.s.setprio(i16)
declare void @llvm.amdgcn.s.setprio.mask(i16, i32)
declare float @llvm.amdgcn.rcp.f32(float)
declare <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float, float, <4 x float>, i32, i32, i32)

; GCN-LABEL: {{^}}test_mask0_blocks_salu:
; GCN: s_add_i32
; GCN-NEXT:      s_setprio 1
; GCN-NEXT: s_add_i32
; GCN-NEXT: s_add_i32
define amdgpu_cs void @test_mask0_blocks_salu(ptr addrspace(1) %out, i32 inreg %x, i32 inreg %y) {
  %add1 = add i32 %x, %y
  call void @llvm.amdgcn.s.setprio.mask(i16 1, i32 0)
  %add2 = add i32 %y, %y
  %sum = add i32 %add1, %add2
  store i32 %sum, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}test_mask0_blocks_valu:
; GCN:      v_add_f32_e32 v{{[0-9]+}}, 1.0
; GCN-NEXT: s_setprio 1
; GCN-NEXT: v_add_f32_e32 v{{[0-9]+}}, 2.0
define amdgpu_cs void @test_mask0_blocks_valu(ptr addrspace(1) %out, float %x, float %y) {
  %add1 = fadd float %x, 1.0
  call void @llvm.amdgcn.s.setprio.mask(i16 1, i32 0)
  %add2 = fadd float %y, 2.0
  %sum = fadd float %add1, %add2
  store float %sum, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}test_mask0_blocks_trans:
; GCN:      v_rcp_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}
; GCN-NEXT: s_setprio 1
; GCN-NEXT: v_rcp_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_cs void @test_mask0_blocks_trans(ptr addrspace(1) %out, float %x, float %y) {
  %rcp1 = call float @llvm.amdgcn.rcp.f32(float %x)
  call void @llvm.amdgcn.s.setprio.mask(i16 1, i32 0)
  %rcp2 = call float @llvm.amdgcn.rcp.f32(float %y)
  %sum = fadd float %rcp1, %rcp2
  store float %sum, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}test_mask4_allows_salu:
; GCN:      s_add_i32
; GCN-NEXT: s_add_i32
; GCN-NEXT: s_add_i32
; GCN-NEXT: s_setprio 1
define amdgpu_cs void @test_mask4_allows_salu(ptr addrspace(1) %out, i32 inreg %x, i32 inreg %y) {
  %add1 = add i32 %x, %y
  call void @llvm.amdgcn.s.setprio.mask(i16 1, i32 4)
  %add2 = add i32 %y, %y
  %sum = add i32 %add1, %add2
  store i32 %sum, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}test_mask4_blocks_valu:
; GCN:      v_add_f32_e32 v{{[0-9]+}}, 1.0
; GCN-NEXT: s_setprio 1
; GCN-NEXT: v_add_f32_e32 v{{[0-9]+}}, 2.0
define amdgpu_cs void @test_mask4_blocks_valu(ptr addrspace(1) %out, float %x, float %y) {
  %add1 = fadd float %x, 1.0
  call void @llvm.amdgcn.s.setprio.mask(i16 1, i32 4)
  %add2 = fadd float %y, 2.0
  %sum = fadd float %add1, %add2
  store float %sum, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}test_mask2_allows_valu:
; GCN:      v_add_f32_e32
; GCN-NEXT: v_add_f32_e32
; GCN-NEXT: v_add_f32_e32
; GCN:      s_setprio 1
define amdgpu_cs void @test_mask2_allows_valu(ptr addrspace(1) %out, float %x, float %y) {
  %add1 = fadd float %x, 1.0
  call void @llvm.amdgcn.s.setprio.mask(i16 1, i32 2)
  %add2 = fadd float %y, 2.0
  %sum = fadd float %add1, %add2
  store float %sum, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}test_mask2_blocks_salu:
; GCN: s_add_i32
; GCN-NEXT: s_setprio 1
; GCN-NEXT: s_add_i32
; GCN-NEXT: s_add_i32
define amdgpu_cs void @test_mask2_blocks_salu(ptr addrspace(1) %out, i32 inreg %x, i32 inreg %y) {
  %add1 = add i32 %x, %y
  call void @llvm.amdgcn.s.setprio.mask(i16 1, i32 2)
  %add2 = add i32 %y, %y
  %sum = add i32 %add1, %add2
  store i32 %sum, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}test_mask1_allows_salu:
; GCN:      s_add_i32
; GCN-NEXT: s_add_i32
; GCN-NEXT: s_add_i32
; GCN:      s_setprio 1
define amdgpu_cs void @test_mask1_allows_salu(ptr addrspace(1) %out, i32 inreg %x, i32 inreg %y) {
  %add1 = add i32 %x, %y
  call void @llvm.amdgcn.s.setprio.mask(i16 1, i32 1)
  %add2 = add i32 %y, %y
  %sum = add i32 %add1, %add2
  store i32 %sum, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}test_mask1_allows_valu:
; GCN:      v_add_f32_e32
; GCN-NEXT: v_add_f32_e32
; GCN-NEXT: v_add_f32_e32
; GCN:      s_setprio 1
define amdgpu_cs void @test_mask1_allows_valu(ptr addrspace(1) %out, float %x, float %y) {
  %add1 = fadd float %x, 1.0
  call void @llvm.amdgcn.s.setprio.mask(i16 1, i32 1)
  %add2 = fadd float %y, 2.0
  %sum = fadd float %add1, %add2
  store float %sum, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}test_mask1024_allows_trans:
; GCN:      v_rcp_f32_e32
; GCN-NEXT: v_rcp_f32_e32
; GCN:      s_setprio 1
define amdgpu_cs void @test_mask1024_allows_trans(ptr addrspace(1) %out, float %x, float %y) {
  %rcp1 = call float @llvm.amdgcn.rcp.f32(float %x)
  call void @llvm.amdgcn.s.setprio.mask(i16 1, i32 1024)
  %rcp2 = call float @llvm.amdgcn.rcp.f32(float %y)
  %sum = fadd float %rcp1, %rcp2
  store float %sum, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}test_mask1024_blocks_salu:
; GCN: s_add_i32
; GCN-NEXT: s_setprio 1
; GCN-NEXT: s_add_i32
; GCN-NEXT: s_add_i32
define amdgpu_cs void @test_mask1024_blocks_salu(ptr addrspace(1) %out, i32 inreg %x, i32 inreg %y) {
  %add1 = add i32 %x, %y
  call void @llvm.amdgcn.s.setprio.mask(i16 1, i32 1024)
  %add2 = add i32 %y, %y
  %sum = add i32 %add1, %add2
  store i32 %sum, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}test_mask6_allows_salu_and_valu:
; GCN:      s_add_i32
; GCN-NEXT: s_add_i32
; GCN-NEXT: s_add_i32
; GCN:      s_setprio 1
define amdgpu_cs void @test_mask6_allows_salu_and_valu(ptr addrspace(1) %out, i32 inreg %x, i32 inreg %y) {
  %add1 = add i32 %x, %y
  call void @llvm.amdgcn.s.setprio.mask(i16 1, i32 6)
  %add2 = add i32 %y, %y
  %sum = add i32 %add1, %add2
  store i32 %sum, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}test_mask8_allows_mfma:
; GCN:      s_setprio 1
; GCN:      v_mfma_f32_4x4x1f32
; GCN:      v_mfma_f32_4x4x1f32
define amdgpu_cs void @test_mask8_allows_mfma(ptr addrspace(1) %out, <4 x float> %in) {
  %mfma1 = call <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float 1.0, float 2.0, <4 x float> %in, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.s.setprio.mask(i16 1, i32 8)
  %mfma2 = call <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float 3.0, float 4.0, <4 x float> %mfma1, i32 0, i32 0, i32 0)
  store <4 x float> %mfma2, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}test_mask0_blocks_mfma:
; GCN:      v_mfma_f32_4x4x1f32
; GCN:      s_setprio 1
; GCN:      v_mfma_f32_4x4x1f32
define amdgpu_cs void @test_mask0_blocks_mfma(ptr addrspace(1) %out, <4 x float> %in) {
  %mfma1 = call <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float 1.0, float 2.0, <4 x float> %in, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.s.setprio.mask(i16 1, i32 0)
  %mfma2 = call <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float 3.0, float 4.0, <4 x float> %mfma1, i32 0, i32 0, i32 0)
  store <4 x float> %mfma2, ptr addrspace(1) %out
  ret void
}

;--- vmem.ll
declare void @llvm.amdgcn.s.setprio.mask(i16, i32)

define amdgpu_cs void @test_invalid_vmem() {
  call void @llvm.amdgcn.s.setprio.mask(i16 1, i32 16)
  ret void
}

;--- vmem-read.ll
declare void @llvm.amdgcn.s.setprio.mask(i16, i32)

define amdgpu_cs void @test_invalid_vmem_read() {
  call void @llvm.amdgcn.s.setprio.mask(i16 1, i32 32)
  ret void
}

;--- vmem-write.ll
declare void @llvm.amdgcn.s.setprio.mask(i16, i32)

define amdgpu_cs void @test_invalid_vmem_write() {
  call void @llvm.amdgcn.s.setprio.mask(i16 1, i32 64)
  ret void
}

;--- ds.ll
declare void @llvm.amdgcn.s.setprio.mask(i16, i32)

define amdgpu_cs void @test_invalid_ds() {
  call void @llvm.amdgcn.s.setprio.mask(i16 1, i32 128)
  ret void
}

;--- ds-read.ll
declare void @llvm.amdgcn.s.setprio.mask(i16, i32)

define amdgpu_cs void @test_invalid_ds_read() {
  call void @llvm.amdgcn.s.setprio.mask(i16 1, i32 256)
  ret void
}

;--- ds-write.ll
declare void @llvm.amdgcn.s.setprio.mask(i16, i32)

define amdgpu_cs void @test_invalid_ds_write() {
  call void @llvm.amdgcn.s.setprio.mask(i16 1, i32 512)
  ret void
}

;--- ldsdma.ll
declare void @llvm.amdgcn.s.setprio.mask(i16, i32)

define amdgpu_cs void @test_invalid_ldsdma() {
  call void @llvm.amdgcn.s.setprio.mask(i16 1, i32 2048)
  ret void
}
