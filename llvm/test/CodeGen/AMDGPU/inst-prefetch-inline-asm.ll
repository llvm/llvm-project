;; Verify that inline assembly is correctly accounted for in the
;; inst_pref_size calculation. The inst_pref_size is computed via MCExpr
;; label subtraction (.Lfunc_end - func_sym), giving exact code size.
;; See inst-prefetch-hint.ll for explanation of the instprefsize expression.

; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 < %s | FileCheck --check-prefix=ASM %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -filetype=obj < %s -o %t.o
; RUN: llvm-objdump -s -j .rodata %t.o | FileCheck --check-prefix=OBJ %s
; RUN: llvm-readobj --symbols %t.o | FileCheck --check-prefix=SYM %s

;; --- .fill directive: .fill 256, 4, 0 => 1024 bytes + 4 (s_endpgm) = 1028 ---
;; pref_size = divideCeil(1028, 128) = 9

; ASM-LABEL: .amdhsa_kernel test_fill
; ASM: .amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-test_fill, 6)<<4)&1008)>>4
; SYM:      Name: test_fill
; SYM-NEXT: Value:
; SYM-NEXT: Size: 1028
;; Object: kernel descriptor at 0x00, COMPUTE_PGM_RSRC3 at 0x2C: pref_size=9 -> 9<<4 = 0x90
; OBJ: 0020 {{.*}}90000000

define amdgpu_kernel void @test_fill() {
  call void asm sideeffect ".fill 256, 4, 0", ""()
  ret void
}

;; --- .space directive: .space 1024 => 1024 bytes + 4 = 1028 ---
;; pref_size = 9

; ASM-LABEL: .amdhsa_kernel test_space
; ASM: .amdhsa_inst_pref_size ((instprefsize(.Lfunc_end1-test_space, 6)<<4)&1008)>>4
; SYM:      Name: test_space
; SYM-NEXT: Value:
; SYM-NEXT: Size: 1028
;; Object: kernel descriptor at 0x40, COMPUTE_PGM_RSRC3 at 0x6C: pref_size=9 -> 9<<4 = 0x90
; OBJ: 0060 {{.*}}90000000

define amdgpu_kernel void @test_space() {
  call void asm sideeffect ".space 1024", ""()
  ret void
}

;; --- Instructions: 32 x s_nop (4 bytes each) = 128 + 4 = 132 ---
;; pref_size = divideCeil(132, 128) = 2

; ASM-LABEL: .amdhsa_kernel test_instructions
; ASM: .amdhsa_inst_pref_size ((instprefsize(.Lfunc_end2-test_instructions, 6)<<4)&1008)>>4
; SYM:      Name: test_instructions
; SYM-NEXT: Value:
; SYM-NEXT: Size: 132
;; Object: kernel descriptor at 0x80, COMPUTE_PGM_RSRC3 at 0xAC: pref_size=2 -> 2<<4 = 0x20
; OBJ: 00a0 {{.*}}20000000

define amdgpu_kernel void @test_instructions() {
  call void asm sideeffect "s_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0", ""()
  ret void
}

;; --- Comments emit no bytes: only s_endpgm = 4 bytes ---
;; pref_size = 1

; ASM-LABEL: .amdhsa_kernel test_comments
; ASM: .amdhsa_inst_pref_size ((instprefsize(.Lfunc_end3-test_comments, 6)<<4)&1008)>>4
; SYM:      Name: test_comments
; SYM-NEXT: Value:
; SYM-NEXT: Size: 4
;; Object: kernel descriptor at 0xC0, COMPUTE_PGM_RSRC3 at 0xEC: pref_size=1 -> 1<<4 = 0x10
; OBJ: 00e0 {{.*}}10000000

define amdgpu_kernel void @test_comments() {
  call void asm sideeffect "; comment 1\0A; comment 2\0A; comment 3", ""()
  ret void
}

;; --- Empty inline asm: only s_endpgm = 4 bytes ---
;; pref_size = 1

; ASM-LABEL: .amdhsa_kernel test_empty_asm
; ASM: .amdhsa_inst_pref_size ((instprefsize(.Lfunc_end4-test_empty_asm, 6)<<4)&1008)>>4
; SYM:      Name: test_empty_asm
; SYM-NEXT: Value:
; SYM-NEXT: Size: 4
;; Object: kernel descriptor at 0x100, COMPUTE_PGM_RSRC3 at 0x12C: pref_size=1 -> 1<<4 = 0x10
; OBJ: 0120 {{.*}}10000000

define amdgpu_kernel void @test_empty_asm() {
  call void asm sideeffect "", ""()
  ret void
}
