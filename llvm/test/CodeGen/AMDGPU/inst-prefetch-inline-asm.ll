; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 < %s | FileCheck %s

;; Test that inline assembly is properly accounted for in
;; amdhsa_inst_pref_size calculation. Each kernel exercises a specific
;; directive or pattern handled by SIInstrInfo::getInlineAsmLength()
;; in lower-bound mode.

; --- .fill directive: .fill count, size, value => count * size bytes ---

; CHECK-LABEL: .amdhsa_kernel test_fill
; CHECK: .amdhsa_inst_pref_size 9
; CHECK: codeLenInByte = 1028
define amdgpu_kernel void @test_fill() {
  call void asm sideeffect ".fill 256, 4, 0", ""()
  ret void
}

; --- .fill with size=1: .fill 1024, 1, 0 => 1024 bytes ---

; CHECK-LABEL: .amdhsa_kernel test_fill_size1
; CHECK: .amdhsa_inst_pref_size 9
; CHECK: codeLenInByte = 1028
define amdgpu_kernel void @test_fill_size1() {
  call void asm sideeffect ".fill 1024, 1, 0", ""()
  ret void
}

; --- .space directive: .space N => N bytes ---

; CHECK-LABEL: .amdhsa_kernel test_space
; CHECK: .amdhsa_inst_pref_size 9
; CHECK: codeLenInByte = 1028
define amdgpu_kernel void @test_space() {
  call void asm sideeffect ".space 1024", ""()
  ret void
}

; --- .zero directive: .zero N => N zero bytes ---

; CHECK-LABEL: .amdhsa_kernel test_zero
; CHECK: .amdhsa_inst_pref_size 9
; CHECK: codeLenInByte = 1028
define amdgpu_kernel void @test_zero() {
  call void asm sideeffect ".zero 1024", ""()
  ret void
}

; --- Data directives: .byte (1) + .short (2) + .long (4) + .quad (8) = 15 bytes ---
;; Lower bound estimate: 15 + 4 (s_endpgm) = 19

; CHECK-LABEL: .amdhsa_kernel test_data_directives
; CHECK: .amdhsa_inst_pref_size 1
; CHECK: codeLenInByte = 19
define amdgpu_kernel void @test_data_directives() {
  call void asm sideeffect ".byte 0\0A.short 0\0A.long 0\0A.quad 0", ""()
  ret void
}

; --- .hword and .word aliases: .hword (2) + .word (4) = 6 bytes ---

; CHECK-LABEL: .amdhsa_kernel test_data_aliases
; CHECK: .amdhsa_inst_pref_size 1
; CHECK: codeLenInByte = 10
define amdgpu_kernel void @test_data_aliases() {
  call void asm sideeffect ".hword 0\0A.word 0", ""()
  ret void
}

; --- Instructions: s_nop lines counted at MinInstAlignment (4 bytes) each ---

; CHECK-LABEL: .amdhsa_kernel test_instructions
; CHECK: .amdhsa_inst_pref_size 8
; CHECK: codeLenInByte = 1020
define amdgpu_kernel void @test_instructions() {
  call void asm sideeffect "s_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0\0As_nop 0", ""()
  ret void
}

; --- Comments should not contribute to size ---
;; Only s_endpgm (4 bytes) — same as an empty kernel.

; CHECK-LABEL: .amdhsa_kernel test_comments
; CHECK: .amdhsa_inst_pref_size 1
; CHECK: codeLenInByte = 4
define amdgpu_kernel void @test_comments() {
  call void asm sideeffect "; comment 1\0A; comment 2\0A; comment 3", ""()
  ret void
}

; --- Labels should not contribute to size ---

; CHECK-LABEL: .amdhsa_kernel test_labels
; CHECK: .amdhsa_inst_pref_size 1
; CHECK: codeLenInByte = 8
define amdgpu_kernel void @test_labels() {
  call void asm sideeffect "my_label:\0As_nop 0", ""()
  ret void
}

; --- Non-emitting directives (.set) should not contribute to size ---
;; Only s_endpgm (4 bytes) — same as an empty kernel.

; CHECK-LABEL: .amdhsa_kernel test_non_emitting
; CHECK: .amdhsa_inst_pref_size 1
; CHECK: codeLenInByte = 4
define amdgpu_kernel void @test_non_emitting() {
  call void asm sideeffect ".set my_const, 42", ""()
  ret void
}

; --- Mixed: .fill (256) + s_nop (4) + comment (0) + .space (128) + s_nop (4) ---
;; Lower bound estimate: 256 + 4 + 0 + 128 + 4 + 4 (s_endpgm) = 396

; CHECK-LABEL: .amdhsa_kernel test_mixed
; CHECK: .amdhsa_inst_pref_size 4
; CHECK: codeLenInByte = 396
define amdgpu_kernel void @test_mixed() {
  call void asm sideeffect ".fill 64, 4, 0\0As_nop 0\0A; comment\0A.space 128\0As_nop 0", ""()
  ret void
}
