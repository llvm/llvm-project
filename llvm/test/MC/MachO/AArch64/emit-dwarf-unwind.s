// RUN: rm -rf %t; mkdir %t
// RUN: llvm-mc -triple x86_64-apple-macos11.0 %s -filetype=obj -o %t/x86_64.o
// RUN: llvm-objdump --macho --dwarf=frames %t/x86_64.o | FileCheck %s --check-prefix TWO-FDES
// RUN: llvm-mc -triple arm64-apple-macos11.0 %s -filetype=obj -o %t/arm64.o
// RUN: llvm-objdump --macho --dwarf=frames %t/arm64.o | FileCheck %s --check-prefix ONE-FDE
// RUN: llvm-mc -triple x86_64-apple-macos11.0 %s -filetype=obj --emit-dwarf-unwind no-compact-unwind -o %t/x86_64-no-dwarf.o
// RUN: llvm-objdump --macho --dwarf=frames %t/x86_64-no-dwarf.o | FileCheck %s --check-prefix ONE-FDE
// RUN: llvm-mc -triple arm64-apple-macos11.0 %s -filetype=obj --emit-dwarf-unwind always -o %t/arm64-dwarf.o
// RUN: llvm-objdump --macho --dwarf=frames %t/arm64-dwarf.o | FileCheck %s --check-prefix TWO-FDES

// TWO-FDES: FDE
// TWO-FDES: FDE

// ONE-FDE-NOT: FDE
// ONE-FDE:     FDE
// ONE-FDE-NOT: FDE

_main:
  .cfi_startproc
  .cfi_def_cfa_offset 16
  ret
  .cfi_endproc

_foo:
  .cfi_startproc
  .cfi_def_cfa_offset 16
  /// This encodes DW_CFA_GNU_args_size which cannot be expressed using compact
  /// unwind, so we must use DWARf unwind for this function.
  .cfi_escape 0x2e, 0x10
  ret
  .cfi_endproc

.subsections_via_symbols
