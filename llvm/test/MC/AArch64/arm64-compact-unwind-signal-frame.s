// RUN: llvm-mc -triple=arm64-apple-ios -filetype=obj %s -o %t.o
// RUN: llvm-objdump --macho --unwind-info --dwarf=frames %t.o | FileCheck %s

/// Signal frames cannot be encoded in compact unwind.  Verify that
/// .cfi_signal_frame causes UNWIND_ARM64_MODE_DWARF (0x03000000) to be
/// emitted in the __compact_unwind section, and that the corresponding DWARF
/// CIE carries the 'S' augmentation character.  A non-signal frame in the
/// same translation unit is encoded normally (UNWIND_ARM64_MODE_FRAMELESS,
/// 0x02000000) and needs no DWARF FDE.

  .globl _f
_f:
  .cfi_startproc
  .cfi_signal_frame
  ret
  .cfi_endproc

  .globl _g
_g:
  .cfi_startproc
  ret
  .cfi_endproc

// CHECK: Contents of __compact_unwind section:
// CHECK:   Entry at offset 0x0:
// CHECK:     compact encoding:     0x03000000
// CHECK:   Entry at offset 0x20:
// CHECK:     start:                0x{{.*}} _g
// CHECK:     compact encoding:     0x02000000

// CHECK: .eh_frame contents:
// CHECK: CIE
// CHECK:   Augmentation:          "zRS"
