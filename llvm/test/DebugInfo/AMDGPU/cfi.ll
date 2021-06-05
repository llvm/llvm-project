; RUN: llc -mcpu=gfx900 -mtriple=amdgcn-amd-amdhsa -filetype=obj -o - %s | llvm-dwarfdump -debug-frame - | FileCheck %s

; CHECK: .debug_frame contents:
; CHECK: 00000000 0000001c ffffffff CIE
; CHECK-NEXT:   Format:                DWARF32
; CHECK-NEXT:   Version:               4
; CHECK-NEXT:   Augmentation:          "[llvm:v0.0]"
; CHECK-NEXT:   Address size:          8
; CHECK-NEXT:   Segment desc size:     0
; CHECK-NEXT:   Code alignment factor: 4
; CHECK-NEXT:   Data alignment factor: 4
; CHECK-NEXT:   Return address column: 16
; CHECK-EMPTY:
; CHECK:   DW_CFA_nop:
; CHECK:   DW_CFA_nop:
; CHECK:   DW_CFA_nop:
; CHECK:   DW_CFA_nop:
; CHECK:   DW_CFA_nop:
; CHECK:   DW_CFA_nop:
; CHECK-EMPTY:
; CHECK: 00000020 {{[0-9]+}} 00000000 FDE cie=00000000 pc=00000000...{{[0-9]+}}
; CHECK-NEXT: Format:       DWARF32
; CHECK-NEXT: DW_CFA_LLVM_def_aspace_cfa: SGPR32_LO16 +0 in addrspace6
; CHECK-NEXT: DW_CFA_expression: PC_REG DW_OP_regx SGPR30_LO16, DW_OP_piece 0x4, DW_OP_regx SGPR31_LO16, DW_OP_piece 0x4
; CHECK-NEXT: DW_CFA_nop:
; CHECK-EMPTY:
; CHECK: .eh_frame contents:
; CHECK-NOT: CIE

define void @func() #0 {
  ret void
}

attributes #0 = { nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 7, !"Dwarf Version", i32 5}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, emissionKind: FullDebug)
!3 = !DIFile(filename: "file", directory: "dir")
