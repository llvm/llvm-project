;; Verify that inline assembly is correctly accounted for in the
;; inst_pref_size calculation. The inst_pref_size is computed via MCExpr
;; label subtraction (.Lfunc_end - func_sym), giving exact code size.
;; See inst-prefetch-hint.ll for explanation of the instprefsize expression.

; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 < %s | FileCheck --check-prefix=GFX11 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1200 < %s | FileCheck --check-prefix=GFX12 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -filetype=obj < %s -o %t.gfx11.o
; RUN: llvm-objdump -s -j .rodata %t.gfx11.o | FileCheck --check-prefix=OBJ-GFX11 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1200 -filetype=obj < %s -o %t.gfx12.o
; RUN: llvm-objdump -s -j .rodata %t.gfx12.o | FileCheck --check-prefix=OBJ-GFX12 %s

;; --- .fill directive: .fill 256, 4, 0 => 1024 bytes + 4 (s_endpgm) = 1028 ---
;; pref_size = divideCeil(1028, 128) = 9

; GFX11-LABEL: .amdhsa_kernel test_fill
; GFX11: .amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-test_fill)<<4)&1008)>>4
; GFX12-LABEL: .amdhsa_kernel test_fill
; GFX12: .amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-test_fill)<<4)&4080)>>4
;; Object: kernel descriptor at 0x00, COMPUTE_PGM_RSRC3 at 0x2C:
;; pref_size=9 -> 9<<4 = 0x90
; OBJ-GFX11: 0020 {{.*}}90000000
; OBJ-GFX12: 0020 {{.*}}90000000

define amdgpu_kernel void @test_fill() {
  call void asm sideeffect ".fill 256, 4, 0", ""()
  ret void
}

;; --- .space directive: .space 1024 => 1024 bytes + 4 = 1028 ---
;; pref_size = 9

; GFX11-LABEL: .amdhsa_kernel test_space
; GFX11: .amdhsa_inst_pref_size ((instprefsize(.Lfunc_end1-test_space)<<4)&1008)>>4
; GFX12-LABEL: .amdhsa_kernel test_space
; GFX12: .amdhsa_inst_pref_size ((instprefsize(.Lfunc_end1-test_space)<<4)&4080)>>4
;; Object: kernel descriptor at 0x40, COMPUTE_PGM_RSRC3 at 0x6C:
;; pref_size=9 -> 9<<4 = 0x90
; OBJ-GFX11: 0060 {{.*}}90000000
; OBJ-GFX12: 0060 {{.*}}90000000

define amdgpu_kernel void @test_space() {
  call void asm sideeffect ".space 1024", ""()
  ret void
}

;; --- Instructions: 32 x s_nop (4 bytes each) = 128 + 4 = 132 ---
;; pref_size = divideCeil(132, 128) = 2

; GFX11-LABEL: .amdhsa_kernel test_instructions
; GFX11: .amdhsa_inst_pref_size ((instprefsize(.Lfunc_end2-test_instructions)<<4)&1008)>>4
; GFX12-LABEL: .amdhsa_kernel test_instructions
; GFX12: .amdhsa_inst_pref_size ((instprefsize(.Lfunc_end2-test_instructions)<<4)&4080)>>4
;; Object: kernel descriptor at 0x80, COMPUTE_PGM_RSRC3 at 0xAC:
;; pref_size=2 -> 2<<4 = 0x20
; OBJ-GFX11: 00a0 {{.*}}20000000
; OBJ-GFX12: 00a0 {{.*}}20000000

define amdgpu_kernel void @test_instructions() {
  call void asm sideeffect "s_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0", ""()
  ret void
}

;; --- Comments emit no bytes: only s_endpgm = 4 bytes ---
;; pref_size = 1

; GFX11-LABEL: .amdhsa_kernel test_comments
; GFX11: .amdhsa_inst_pref_size ((instprefsize(.Lfunc_end3-test_comments)<<4)&1008)>>4
; GFX12-LABEL: .amdhsa_kernel test_comments
; GFX12: .amdhsa_inst_pref_size ((instprefsize(.Lfunc_end3-test_comments)<<4)&4080)>>4
;; Object: kernel descriptor at 0xC0, COMPUTE_PGM_RSRC3 at 0xEC:
;; pref_size=1 -> 1<<4 = 0x10
; OBJ-GFX11: 00e0 {{.*}}10000000
; OBJ-GFX12: 00e0 {{.*}}10000000

define amdgpu_kernel void @test_comments() {
  call void asm sideeffect "; comment 1\0A; comment 2\0A; comment 3", ""()
  ret void
}

;; --- Empty inline asm: only s_endpgm = 4 bytes ---
;; pref_size = 1

; GFX11-LABEL: .amdhsa_kernel test_empty_asm
; GFX11: .amdhsa_inst_pref_size ((instprefsize(.Lfunc_end4-test_empty_asm)<<4)&1008)>>4
; GFX12-LABEL: .amdhsa_kernel test_empty_asm
; GFX12: .amdhsa_inst_pref_size ((instprefsize(.Lfunc_end4-test_empty_asm)<<4)&4080)>>4
;; Object: kernel descriptor at 0x100, COMPUTE_PGM_RSRC3 at 0x12C:
;; pref_size=1 -> 1<<4 = 0x10
; OBJ-GFX11: 0120 {{.*}}10000000
; OBJ-GFX12: 0120 {{.*}}10000000

define amdgpu_kernel void @test_empty_asm() {
  call void asm sideeffect "", ""()
  ret void
}

;; --- Multiple inline asm blocks: .fill (512) + .space (512) + s_endpgm (4) = 1028 ---
;; pref_size = divideCeil(1028, 128) = 9

; GFX11-LABEL: .amdhsa_kernel test_multiple_asm
; GFX11: .amdhsa_inst_pref_size ((instprefsize(.Lfunc_end5-test_multiple_asm)<<4)&1008)>>4
; GFX12-LABEL: .amdhsa_kernel test_multiple_asm
; GFX12: .amdhsa_inst_pref_size ((instprefsize(.Lfunc_end5-test_multiple_asm)<<4)&4080)>>4
;; Object: kernel descriptor at 0x140, COMPUTE_PGM_RSRC3 at 0x16C:
;; pref_size=9 -> 9<<4 = 0x90
; OBJ-GFX11: 0160 {{.*}}90000000
; OBJ-GFX12: 0160 {{.*}}90000000

define amdgpu_kernel void @test_multiple_asm() {
  call void asm sideeffect ".fill 128, 4, 0", ""()
  call void asm sideeffect ".space 512", ""()
  ret void
}

;; --- Large function that exceeds GFX11 6-bit field max (63) ---
;; .fill 2048, 4, 0 = 8192 bytes + 4 = 8196 bytes
;; divideCeil(8196, 128) = 65, but GFX11 max = (1<<6)-1 = 63
;; pref_size should clamp to 63

; GFX11-LABEL: .amdhsa_kernel test_clamping
; GFX11: .amdhsa_inst_pref_size ((instprefsize(.Lfunc_end6-test_clamping)<<4)&1008)>>4
; GFX12-LABEL: .amdhsa_kernel test_clamping
; GFX12: .amdhsa_inst_pref_size ((instprefsize(.Lfunc_end6-test_clamping)<<4)&4080)>>4
;; Object: kernel descriptor at 0x180, COMPUTE_PGM_RSRC3 at 0x1AC:
;; gfx11: clamped to 63 -> 63<<4 = 0x3F0
;; gfx12: no clamping, 65 -> 65<<4 = 0x410
; OBJ-GFX11: 01a0 {{.*}}f0030000
; OBJ-GFX12: 01a0 {{.*}}10040000

define amdgpu_kernel void @test_clamping() {
  call void asm sideeffect ".fill 2048, 4, 0", ""()
  ret void
}

;; --- Large function that exceeds both GFX11 and GFX12 field max ---
;; .fill 8192, 4, 0 = 32768 bytes + 4 = 32772 bytes
;; divideCeil(32772, 128) = 257
;; GFX11 max = 63, GFX12 max = 255 -> both clamp

; GFX11-LABEL: .amdhsa_kernel test_clamping_both
; GFX11: .amdhsa_inst_pref_size ((instprefsize(.Lfunc_end7-test_clamping_both)<<4)&1008)>>4
; GFX12-LABEL: .amdhsa_kernel test_clamping_both
; GFX12: .amdhsa_inst_pref_size ((instprefsize(.Lfunc_end7-test_clamping_both)<<4)&4080)>>4
;; Object: kernel descriptor at 0x1C0, COMPUTE_PGM_RSRC3 at 0x1EC:
;; gfx11: clamped to 63 -> 63<<4 = 0x3F0
;; gfx12: clamped to 255 -> 255<<4 = 0xFF0
; OBJ-GFX11: 01e0 {{.*}}f0030000
; OBJ-GFX12: 01e0 {{.*}}f00f0000

define amdgpu_kernel void @test_clamping_both() {
  call void asm sideeffect ".fill 8192, 4, 0", ""()
  ret void
}
