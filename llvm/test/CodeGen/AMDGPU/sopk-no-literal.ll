; REQUIRES: asserts
; RUN: llc -O0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -debug-only=branch-relaxation -verify-machineinstrs < %s 2>&1 | FileCheck --check-prefix=GFX10 %s

; GFX10: Basic blocks after relaxation
; GFX10: %bb.0	offset=00000000	size=0x1c

; Each instruction in the following kernel is 4 bytes in size,
; except s_load_b32 which is 8 bytes in size. Hence, 0x1c bytes in total.
define amdgpu_kernel void @test_sopk_size(i32 %var.mode) {
; GFX10-LABEL: test_sopk_size:
; GFX10:  ; %bb.0:
; GFX10:    s_load_b32 s0, s[2:3], 0x0
; GFX10:    s_mov_b32 s1, 3
; GFX10:    s_setreg_b32 hwreg(HW_REG_MODE, 0, 2), s1
; GFX10:    s_waitcnt lgkmcnt(0)
; GFX10:    s_setreg_b32 hwreg(HW_REG_MODE, 0, 3), s0
; GFX10:    s_endpgm
  call void @llvm.amdgcn.s.setreg(i32 2049, i32 3)
  call void @llvm.amdgcn.s.setreg(i32 4097, i32 %var.mode)
  call void asm sideeffect "", ""()
  ret void
}

declare void @llvm.amdgcn.s.setreg(i32 immarg, i32)
