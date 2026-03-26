// RUN: llvm-mc -triple=armv7k-apple-ios -filetype=obj %s -o %t.o
// RUN: llvm-objdump --macho --unwind-info --dwarf=frames %t.o | FileCheck %s

/// Signal frames cannot be encoded in compact unwind.  Verify that
/// .cfi_signal_frame causes UNWIND_ARM_MODE_DWARF (0x04000000) to be emitted
/// in the __compact_unwind section, and that the corresponding DWARF CIE
/// carries the 'S' augmentation character.

  .globl _f
_f:
  .cfi_startproc
  .cfi_signal_frame
  bx lr
  .cfi_endproc

// CHECK: Contents of __compact_unwind section:
// CHECK:   Entry at offset 0x0:
// CHECK:     start:                0x0 _f
// CHECK:     length:               0x4
// CHECK:     compact encoding:     0x04000000

// CHECK: .eh_frame contents:
// CHECK: CIE
// CHECK:   Augmentation:          "zRS"
