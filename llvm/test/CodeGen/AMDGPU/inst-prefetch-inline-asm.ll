;; Verify that inline assembly is correctly accounted for in the
;; inst_pref_size calculation. The inst_pref_size is computed via MCExpr
;; label subtraction (code_end - func_sym), giving exact code size.
;; This resolves at assembly time, so we verify via object file checks.

; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -filetype=obj < %s -o %t.o
; RUN: llvm-objdump -s -j .rodata %t.o | FileCheck --check-prefix=OBJ %s
; RUN: llvm-readobj --symbols %t.o | FileCheck --check-prefix=SYM %s

;; --- .fill directive: .fill 256, 4, 0 => 1024 bytes + 4 (s_endpgm) = 1028 ---
;; pref_size = divideCeil(1028, 128) = 9

; SYM:      Name: test_fill
; SYM-NEXT: Value:
; SYM-NEXT: Size: 1028

define amdgpu_kernel void @test_fill() {
  call void asm sideeffect ".fill 256, 4, 0", ""()
  ret void
}

;; --- .space directive: .space 1024 => 1024 bytes + 4 = 1028 ---
;; pref_size = 9

; SYM:      Name: test_space
; SYM-NEXT: Value:
; SYM-NEXT: Size: 1028

define amdgpu_kernel void @test_space() {
  call void asm sideeffect ".space 1024", ""()
  ret void
}

;; --- Instructions: 32 x s_nop (4 bytes each) = 128 + 4 = 132 ---
;; pref_size = divideCeil(132, 128) = 2

; SYM:      Name: test_instructions
; SYM-NEXT: Value:
; SYM-NEXT: Size: 132

define amdgpu_kernel void @test_instructions() {
  call void asm sideeffect "s_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0", ""()
  ret void
}

;; --- Comments emit no bytes: only s_endpgm = 4 bytes ---
;; pref_size = 1

; SYM:      Name: test_comments
; SYM-NEXT: Value:
; SYM-NEXT: Size: 4

define amdgpu_kernel void @test_comments() {
  call void asm sideeffect "; comment 1\0A; comment 2\0A; comment 3", ""()
  ret void
}

;; --- Empty inline asm: only s_endpgm = 4 bytes ---
;; pref_size = 1

; SYM:      Name: test_empty_asm
; SYM-NEXT: Value:
; SYM-NEXT: Size: 4

define amdgpu_kernel void @test_empty_asm() {
  call void asm sideeffect "", ""()
  ret void
}

;; Object file checks: verify COMPUTE_PGM_RSRC3 at offset 0x2C in each
;; 64-byte kernel descriptor. GFX11 inst_pref_size is bits [9:4].
;;
;; test_fill:         1028 bytes -> pref_size=9  -> 9<<4  = 0x90
;; test_space:        1028 bytes -> pref_size=9  -> 9<<4  = 0x90
;; test_instructions:  132 bytes -> pref_size=2  -> 2<<4  = 0x20
;; test_comments:        4 bytes -> pref_size=1  -> 1<<4  = 0x10
;; test_empty_asm:       4 bytes -> pref_size=1  -> 1<<4  = 0x10

; OBJ: 0020 {{.*}}90000000
; OBJ: 0060 {{.*}}90000000
; OBJ: 00a0 {{.*}}20000000
; OBJ: 00e0 {{.*}}10000000
; OBJ: 0120 {{.*}}10000000
