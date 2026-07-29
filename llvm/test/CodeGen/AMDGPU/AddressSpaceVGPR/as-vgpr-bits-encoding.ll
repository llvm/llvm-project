; RUN: llc -verify-machineinstrs -mtriple=amdgcn -mcpu=gfx1200 -o - %s | llvm-mc -triple=amdgcn -mcpu=gfx1200 -filetype=obj -o - | llvm-objdump -d --mcpu=gfx1200 - | FileCheck --check-prefix=DIS %s
; RUN: llc -verify-machineinstrs -mtriple=amdgcn -mcpu=gfx1250 -o - %s | llvm-mc -triple=amdgcn -mcpu=gfx1250 -filetype=obj -o - | llvm-objdump -d --mcpu=gfx1250 - | FileCheck --check-prefixes=DIS,GFX1250 %s

; The sub-dword accesses of the VGPR "as memory" address space (13) that
; AMDGPULowerIdxOps expands, taken to machine code and back. The other tests here
; check assembly text, which does not show that the operand forms the expansion
; picks can be assembled and encoded: a mask it builds from a known bit offset is
; too wide for an inline constant and becomes a 32-bit literal, and a bit offset
; reaches the bit-field extract as either an inline constant or an SGPR. On a
; subtarget with more than 256 addressable VGPRs the whole-dword accesses the
; expansion creates additionally carry the S_SET_VGPR_MSB encoding, so check
; there too.
;
; The expansion runs after instruction selection and is handed the same pseudos
; by both selectors, which produce identical code for these functions, so one
; selector is enough here. as-vgpr-bits.ll covers the two separately.

; A statically known bit offset turns the mask into a literal operand of the
; bit-field insert.
define amdgpu_ps void @store_i8_aligned(ptr addrspace(13) inreg %p, i32 inreg %vv) {
; DIS-LABEL: <store_i8_aligned>:
; GFX1250:  s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
; DIS:      v_movrels_b32_e32 v1, v0
; DIS:      v_bfi_b32 v0, 0xff, v0, v1
; DIS:      v_movreld_b32_e32 v0, v0
  %v = trunc i32 %vv to i8
  store i8 %v, ptr addrspace(13) %p, align 4
  ret void
}

define amdgpu_ps void @store_i16_aligned(ptr addrspace(13) inreg %p, i32 inreg %vv) {
; DIS-LABEL: <store_i16_aligned>:
; DIS:      v_movrels_b32_e32 v1, v0
; DIS:      v_bfi_b32 v0, 0xffff, v0, v1
; DIS:      v_movreld_b32_e32 v0, v0
  %v = trunc i32 %vv to i16
  store i16 %v, ptr addrspace(13) %p, align 4
  ret void
}

; Without one the mask is computed at run time and is an SGPR instead.
define amdgpu_ps void @store_i8_dynamic_mask(ptr addrspace(13) inreg %p, i32 inreg %vv) {
; DIS-LABEL: <store_i8_dynamic_mask>:
; DIS:      v_movrels_b32_e32 v1, v0
; DIS:      v_bfi_b32 v0, s1, v0, v1
; DIS:      v_movreld_b32_e32 v0, v0
  %v = trunc i32 %vv to i8
  store i8 %v, ptr addrspace(13) %p
  ret void
}

; Both bit offset and width of the extract are inline constants here.
define amdgpu_ps i32 @load_i8_zext_aligned(ptr addrspace(13) inreg %p) {
; DIS-LABEL: <load_i8_zext_aligned>:
; DIS:      v_movrels_b32_e32 v0, v0
; DIS:      v_bfe_u32 v0, v0, 0, 8
  %x = load i8, ptr addrspace(13) %p, align 4
  %z = zext i8 %x to i32
  ret i32 %z
}

; A run-time bit offset is an SGPR, and the sign-extending load selects the
; signed extract.
define amdgpu_ps i32 @load_i8_sext_dynamic(ptr addrspace(13) inreg %p) {
; DIS-LABEL: <load_i8_sext_dynamic>:
; DIS:      v_movrels_b32_e32 v0, v0
; DIS:      v_bfe_i32 v0, v0, s0, 8
  %x = load i8, ptr addrspace(13) %p
  %s = sext i8 %x to i32
  ret i32 %s
}
