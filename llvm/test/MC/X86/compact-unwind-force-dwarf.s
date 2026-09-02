// RUN: llvm-mc -triple x86_64-apple-macos10.6 -filetype=obj --emit-dwarf-unwind dwarf-only %s -o %t.dwarf.o
// RUN: llvm-objdump --macho --unwind-info --dwarf=frames %t.dwarf.o | FileCheck %s --check-prefixes=CHECK,DWARF
// RUN: llvm-mc -triple x86_64-apple-macos10.6 -filetype=obj %s -o %t.compact.o
// RUN: llvm-objdump --macho --unwind-info --dwarf=frames %t.compact.o | FileCheck %s --check-prefixes=CHECK,COMPACT

_f:
  .cfi_startproc
  pushq %rbp
  .cfi_def_cfa_offset 16
  .cfi_offset %rbp, -16
  movq %rsp, %rbp
  .cfi_def_cfa_register %rbp
  popq %rbp
  ret
  .cfi_endproc

// On x86, a compact unwind encoding of 0x04000000 indicates
// "fall back on DWARF unwind"

// CHECK: Contents of __compact_unwind section:
// CHECK:   Entry at offset 0x0:
// CHECK:     start:                0x[[#%x,F:]] _f
// CHECK:     length:               0x6
// COMPACT:   compact encoding:     0x01000000
// DWARF:     compact encoding:     0x04000000

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

// CHECK: FDE cie=00000000 pc=00000000...00000006
// CHECK:   Format:       DWARF32
// CHECK:   DW_CFA_advance_loc: 1 to 0x1
// CHECK:   DW_CFA_def_cfa_offset: +16
// CHECK:   DW_CFA_offset: reg6 -16
// CHECK:   DW_CFA_advance_loc: 3 to 0x4
// CHECK:   DW_CFA_def_cfa_register: reg6

// CHECK:   0x0: CFA=reg7+8: reg16=[CFA-8]
// CHECK:   0x1: CFA=reg7+16: reg6=[CFA-16], reg16=[CFA-8]
// CHECK:   0x4: CFA=reg6+16: reg6=[CFA-16], reg16=[CFA-8]
