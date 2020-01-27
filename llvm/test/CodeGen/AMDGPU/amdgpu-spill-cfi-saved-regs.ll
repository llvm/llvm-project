; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=asm -amdgpu-spill-cfi-saved-regs -o - %s | FileCheck --check-prefixes=CHECK,WAVE64 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -mattr=+wavefrontsize32,-wavefrontsize64 -filetype=asm -amdgpu-spill-cfi-saved-regs -o - %s | FileCheck --check-prefixes=CHECK,WAVE32 %s

; CHECK-LABEL: kern:
; CHECK: .cfi_startproc
; CHECK-NOT: .cfi_{{.*}}
; CHECK: %bb.0:
; CHECK-NEXT: .cfi_escape 0x0f, 0x03, 0x30, 0x36, 0xe1
; CHECK-NEXT: .cfi_undefined 16
; CHECK-NOT: .cfi_{{.*}}
; CHECK: .cfi_endproc
define protected amdgpu_kernel void @kern() #0 {
entry:
  ret void
}

; CHECK-LABEL: func:
; CHECK: .cfi_startproc
; CHECK-NOT: .cfi_{{.*}}
; CHECK: %bb.0:
; SGPR32 = 64
; CHECK-NEXT: .cfi_llvm_def_aspace_cfa 64, 0, 6
; CHECK-NEXT: .cfi_escape 0x10, 0x10, 0x08, 0x90, 0x3e, 0x93, 0x04, 0x90, 0x3f, 0x93, 0x04


; FIXME: ideally this would not care what VGPR we spill to, but since we are
; using .cfi_escape it isn't trivial/possible to make this general yet

; CHECK: v_writelane_b32 v0, s30, 0
; CHECK-NEXT: v_writelane_b32 v0, s31, 1

; DW_CFA_expression [0x10]
;   PC_64 ULEB128(17)=[0x10]
;   BLOCK_LENGTH ULEB128(12)=[0x0c]
;     DW_OP_regx [0x90]
;       VGPR0_wave64 ULEB128(2560)=[0x80, 0x14]
;     DW_OP_bit_piece [0x9d]
;       PIECE_SIZE [0x20]
;       PIECE_OFFSET [0x00]
;     DW_OP_regx [0x90]
;       VGPR0_wave64 ULEB128(2560)=[0x80, 0x14]
;     DW_OP_bit_piece [0x9d]
;       PIECE_SIZE [0x20]
;       PIECE_OFFSET [0x20]
; WAVE64-NEXT: .cfi_escape 0x10, 0x10, 0x0c, 0x90, 0x80, 0x14, 0x9d, 0x20, 0x00, 0x90, 0x80, 0x14, 0x9d, 0x20, 0x20

; DW_CFA_expression [0x10]
;   PC_64 ULEB128(17)=[0x10]
;   BLOCK_LENGTH ULEB128(12)=[0x0c]
;     DW_OP_regx [0x90]
;       VGPR0_wave32 ULEB128(1536)=[0x80, 0x0c]
;     DW_OP_bit_piece [0x9d]
;       PIECE_SIZE [0x20]
;       PIECE_OFFSET [0x00]
;     DW_OP_regx [0x90]
;       VGPR0_wave32 ULEB128(1536)=[0x80, 0x0c]
;     DW_OP_bit_piece [0x9d]
;       PIECE_SIZE [0x20]
;       PIECE_OFFSET [0x20]
; WAVE32-NEXT: .cfi_escape 0x10, 0x10, 0x0c, 0x90, 0x80, 0x0c, 0x9d, 0x20, 0x00, 0x90, 0x80, 0x0c, 0x9d, 0x20, 0x20


; WAVE64: v_writelane_b32 v0, exec_lo, 2
; WAVE64-NEXT: v_writelane_b32 v0, exec_hi, 3
; DW_CFA_expression [0x10]
;   EXEC_MASK_wave64 ULEB128(17)=[0x11]
;   BLOCK_LENGTH ULEB128(12)=[0x0c]
;     DW_OP_regx [0x90]
;       VGPR0_wave64 ULEB128(2560)=[0x80, 0x14]
;     DW_OP_bit_piece [0x9d]
;       PIECE_SIZE [0x20]
;       PIECE_OFFSET [0x40]
;     DW_OP_regx [0x90]
;       VGPR0_wave64 ULEB128(2560)=[0x80, 0x14]
;     DW_OP_bit_piece [0x9d]
;       PIECE_SIZE [0x20]
;       PIECE_OFFSET [0x60]
; WAVE64-NEXT: .cfi_escape 0x10, 0x11, 0x0c, 0x90, 0x80, 0x14, 0x9d, 0x20, 0x40, 0x90, 0x80, 0x14, 0x9d, 0x20, 0x60

; WAVE32: v_writelane_b32 v0, exec_lo, 2
; DW_CFA_expression [0x10]
;   EXEC_MASK_wave32 ULEB128(1)=[0x01]
;   BLOCK_LENGTH ULEB128(6)=[0x06]
;     DW_OP_regx [0x90]
;       VGPR0_wave32 ULEB128(1536)=[0x80, 0x0c]
;     DW_OP_bit_piece [0x9d]
;       PIECE_SIZE [0x20]
;       PIECE_OFFSET [0x40]
; WAVE32-NEXT: .cfi_escape 0x10, 0x01, 0x06, 0x90, 0x80, 0x0c, 0x9d, 0x20, 0x40

; CHECK-NOT: .cfi_{{.*}}
; CHECK: .cfi_endproc
define hidden void @func() #0 {
entry:
  ret void
}

attributes #0 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "filename", directory: "directory")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
