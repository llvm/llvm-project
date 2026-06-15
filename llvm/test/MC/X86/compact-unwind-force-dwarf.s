// RUN: llvm-mc -triple x86_64-apple-macos10.6 -filetype=obj --emit-dwarf-unwind dwarf-only %s -o %t.o
// RUN: llvm-objdump --macho --unwind-info --dwarf=frames %t.o | FileCheck %s

_f:
  .cfi_startproc
  ret
  .cfi_endproc

// CHECK: Contents of __compact_unwind section:
// CHECK:   Entry at offset 0x0:
// CHECK:     start:                0x[[#%x,F:]] _f
// CHECK:     length:               0x1
// CHECK:     compact encoding:     0x04000000

// CHECK: .eh_frame contents:
// CHECK: 00000000 00000014 00000000 CIE
// CHECK:   Format:                DWARF32
// CHECK:   Version:               1
// CHECK:   Augmentation:          "zR"
// CHECK:   Code alignment factor: 1
// CHECK:   Data alignment factor: -8
// CHECK:   Return address column: 16
// CHECK:   Augmentation data:     10

// CHECK:   DW_CFA_def_cfa: reg7 +8
// CHECK:   DW_CFA_offset: reg16 -8
// CHECK:   DW_CFA_nop:
// CHECK:   DW_CFA_nop:

// CHECK:   CFA=reg7+8: reg16=[CFA-8]

// CHECK: FDE cie=00000000 pc=00000000...00000001
// CHECK:   Format:       DWARF32
// CHECK:   DW_CFA_nop:
// CHECK:   DW_CFA_nop:
// CHECK:   DW_CFA_nop:
// CHECK:   DW_CFA_nop:
// CHECK:   DW_CFA_nop:
// CHECK:   DW_CFA_nop:
// CHECK:   DW_CFA_nop:

// CHECK:   0x0: CFA=reg7+8: reg16=[CFA-8]