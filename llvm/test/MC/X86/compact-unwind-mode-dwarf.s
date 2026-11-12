// RUN: llvm-mc -triple x86_64-apple-macos10.6 -filetype=obj %s -o %t.o
// RUN: llvm-objdump --macho --unwind-info --dwarf=frames %t.o | FileCheck %s

/// For functions whose unwind info cannot be encoded with compact unwind, make
/// sure that we encode them using DWARF unwind, and make sure we emit a compact
/// unwind entry that indicates that a DWARF encoding is being used.

_f:
  .cfi_startproc
  ## This encodes DW_CFA_GNU_args_size which cannot be expressed using compact
  ## unwind, so we must use DWARF unwind instead.
  .cfi_escape 0x2e, 0x10
  ret
  .cfi_endproc

_g:
  .cfi_startproc
  ## This encodes DW_CFA_GNU_args_size which cannot be expressed using compact
  ## unwind, so we must use DWARF unwind instead.
  .cfi_escape 0x2e, 0x10
  ret
  .cfi_endproc

// CHECK: Contents of __compact_unwind section:
// CHECK:   Entry at offset 0x0:
// CHECK:     start:                0x[[#%x,F:]] _f
// CHECK:     length:               0x1
// CHECK:     compact encoding:     0x04000000
// CHECK:   Entry at offset 0x20:
// CHECK:     start:                0x[[#%x,G:]] _g
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

// CHECK: FDE cie=00000000 pc=[[#%.8x,F]]...
// CHECK:   Format:       DWARF32
// CHECK:   DW_CFA_GNU_args_size: +16

// CHECK: FDE cie=00000000 pc=[[#%.8x,G]]...
// CHECK:   Format:       DWARF32
// CHECK:   DW_CFA_GNU_args_size: +16
