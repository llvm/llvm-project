// RUN: llvm-mc -triple arm64-apple-macos10.6 -filetype=obj --emit-dwarf-unwind dwarf-only %s -o %t.o
// RUN: llvm-objdump --macho --unwind-info --dwarf=frames %t.o | FileCheck %s

_f:
  .cfi_startproc
  ret
  .cfi_endproc

// CHECK: Contents of __compact_unwind section:
// CHECK:   Entry at offset 0x0:
// CHECK:     start:                0x0 ltmp0
// CHECK:     length:               0x4
// CHECK:     compact encoding:     0x03000000

// CHECK: .eh_frame contents:

// CHECK: 00000000 00000010 00000000 CIE
// CHECK:   Format:                DWARF32
// CHECK:   Version:               1
// CHECK:   Augmentation:          "zR"
// CHECK:   Code alignment factor: 1
// CHECK:   Data alignment factor: -8
// CHECK:   Return address column: 30
// CHECK:   Augmentation data:     10

// CHECK:   DW_CFA_def_cfa: reg31 +0

// CHECK:   CFA=reg31

// CHECK: FDE cie=00000000
// CHECK:   Format:       DWARF32
// CHECK:   DW_CFA_nop:
// CHECK:   DW_CFA_nop:
// CHECK:   DW_CFA_nop:

// CHECK:   CFA=reg31
