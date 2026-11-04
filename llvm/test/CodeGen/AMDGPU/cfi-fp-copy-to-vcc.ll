; RUN: llc -mtriple=amdgpu9.0a-amd-amdhsa < %s \
; RUN:   | FileCheck --check-prefixes=CHECK,WAVE64 --implicit-check-not=4294967295 %s
; RUN: llc -mtriple=amdgpu10.10-amd-amdhsa < %s \
; RUN:   | FileCheck --check-prefixes=CHECK,WAVE32 --implicit-check-not=4294967295 %s
; RUN: llc -mtriple=amdgpu9.0a-amd-amdhsa -filetype=obj < %s \
; RUN:   | llvm-dwarfdump --eh-frame - | FileCheck --check-prefix=DWARF64 %s

; Check the CFI when the frame pointer is saved to a VCC half. VCC_32_LO and
; VCC_32_HI only exist in wave32, so wave64 describes the copy as part of VCC_64.

define void @fp_copy_to_vcc_lo() #0 {
; CHECK-LABEL: fp_copy_to_vcc_lo:
; WAVE64:        s_mov_b32 vcc_lo, s33
; WAVE64-NEXT:   .cfi_llvm_vector_registers 65, 768, 0, 32
; WAVE32:        s_mov_b32 vcc_lo, s33
; WAVE32-NEXT:   .cfi_register 65, 512
; CHECK:         s_mov_b32 s33, s32
; CHECK:         s_mov_b32 s33, vcc_lo
  call void asm sideeffect "; clobber scratch SGPRs",
    "~{s4},~{s5},~{s6},~{s7},~{s8},~{s9},~{s10},~{s11},~{s12},~{s13},~{s14}
    ,~{s15},~{s16},~{s17},~{s18},~{s19},~{s20},~{s21},~{s22},~{s23},~{s24}
    ,~{s25},~{s26},~{s27},~{s28},~{s29},~{s40},~{s41},~{s42},~{s43},~{s44}
    ,~{s45},~{s46},~{s47},~{s56},~{s57},~{s58},~{s59},~{s60},~{s61},~{s62}
    ,~{s63},~{s72},~{s73},~{s74},~{s75},~{s76},~{s77},~{s78},~{s79},~{s88}
    ,~{s89},~{s90},~{s91},~{s92},~{s93},~{s94},~{s95}"()
  ret void
}

define void @fp_copy_to_vcc_hi() #0 {
; CHECK-LABEL: fp_copy_to_vcc_hi:
; WAVE64:        s_mov_b32 vcc_hi, s33
; WAVE64-NEXT:   .cfi_llvm_vector_registers 65, 768, 1, 32
; WAVE32:        s_mov_b32 vcc_hi, s33
; WAVE32-NEXT:   .cfi_register 65, 513
; CHECK:         s_mov_b32 s33, s32
; CHECK:         s_mov_b32 s33, vcc_hi
  call void asm sideeffect "; clobber scratch SGPRs and vcc_lo",
    "~{s4},~{s5},~{s6},~{s7},~{s8},~{s9},~{s10},~{s11},~{s12},~{s13},~{s14}
    ,~{s15},~{s16},~{s17},~{s18},~{s19},~{s20},~{s21},~{s22},~{s23},~{s24}
    ,~{s25},~{s26},~{s27},~{s28},~{s29},~{s40},~{s41},~{s42},~{s43},~{s44}
    ,~{s45},~{s46},~{s47},~{s56},~{s57},~{s58},~{s59},~{s60},~{s61},~{s62}
    ,~{s63},~{s72},~{s73},~{s74},~{s75},~{s76},~{s77},~{s78},~{s79},~{s88}
    ,~{s89},~{s90},~{s91},~{s92},~{s93},~{s94},~{s95},~{vcc_lo}"()
  ret void
}

; DWARF64-LABEL: FDE
; DWARF64:         DW_CFA_expression: SGPR33 DW_OP_regx VCC, DW_OP_LLVM_user DW_OP_LLVM_offset_uconst 0x0
; DWARF64-LABEL: FDE
; DWARF64:         DW_CFA_expression: SGPR33 DW_OP_regx VCC, DW_OP_LLVM_user DW_OP_LLVM_offset_uconst 0x4

attributes #0 = { uwtable "frame-pointer"="all" }
