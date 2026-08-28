// RUN: llvm-mc -triple arm64-apple-macos10.6 -filetype=obj %s -o %t.compact.o
// RUN: llvm-objdump --macho --unwind-info --dwarf=frames %t.compact.o | FileCheck %s --check-prefixes=CHECK,COMPACT
// RUN: llvm-mc -triple arm64-apple-macos10.6 -filetype=obj --emit-dwarf-unwind dwarf-only %s -o %t.dwarf.o
// RUN: llvm-objdump --macho --unwind-info --dwarf=frames %t.dwarf.o | FileCheck %s --check-prefixes=CHECK,DWARF

_f:
  .cfi_startproc
  ret
  .cfi_endproc

// On arm64, a compact unwind encoding of 0x03000000 indicates
// "fall back on DWARF unwind".

// CHECK: Contents of __compact_unwind section:
// CHECK:   Entry at offset 0x0:
// CHECK:     start:                0x0 ltmp0
// CHECK:     length:               0x4
// COMPACT:   compact encoding:     0x02000000
// DWARF:     compact encoding:     0x03000000

// CHECK: .eh_frame contents:

// DWARF: 00000000 00000010 00000000 CIE
// DWARF:   Format:                DWARF32
// DWARF:   Version:               1
// DWARF:   Augmentation:          "zR"
// DWARF:   Code alignment factor: 1
// DWARF:   Data alignment factor: -8
// DWARF:   Return address column: 30
// DWARF:   Augmentation data:     10

// DWARF:   DW_CFA_def_cfa: reg31 +0

// DWARF:   CFA=reg31

// DWARF: FDE cie=00000000
// DWARF:   Format:       DWARF32
// DWARF:   DW_CFA_nop:
// DWARF:   DW_CFA_nop:
// DWARF:   DW_CFA_nop:

// DWARF:   CFA=reg31
