// Test the bits of .eh_frame on mips that are already implemented correctly.

// RUN: llvm-mc -filetype=obj %s -o %t.o -arch=mips
// RUN: llvm-objdump -r -section=.rel.eh_frame %t.o | FileCheck --check-prefix=REL32 %s
// RUN: llvm-dwarfdump -eh-frame %t.o | FileCheck --check-prefix=DWARF32 %s

// RUN: llvm-mc -filetype=obj %s -o %t.o -arch=mipsel
// RUN: llvm-objdump -r -section=.rel.eh_frame %t.o | FileCheck --check-prefix=REL32 %s
// RUN: llvm-dwarfdump -eh-frame %t.o | FileCheck --check-prefix=DWARF32 %s

// RUN: llvm-mc -filetype=obj %s -o %t.o -arch=mips64
// RUN: llvm-objdump -r -section=.rela.eh_frame %t.o | FileCheck --check-prefix=REL64 %s
// RUN: llvm-dwarfdump -eh-frame %t.o | FileCheck --check-prefix=DWARF64 %s

// RUN: llvm-mc -filetype=obj %s -o %t.o -arch=mips64el
// RUN: llvm-objdump -r -section=.rela.eh_frame %t.o | FileCheck --check-prefix=REL64 %s
// RUN: llvm-dwarfdump -eh-frame %t.o | FileCheck --check-prefix=DWARF64 %s

func:
	.cfi_startproc
	.cfi_endproc

// REL32: R_MIPS_32
// REL64: R_MIPS_64/R_MIPS_NONE/R_MIPS_NONE

// DWARF32: 00000000 00000010 ffffffff CIE
// DWARF32:   Version:               1
// DWARF32:   Augmentation:          "zR"
// DWARF32:   Code alignment factor: 1
// DWARF32:   Data alignment factor: -4
// DWARF32:   Return address column: 31
// DWARF32:   Augmentation data:     0B
//                                   ^^ fde pointer encoding: DW_EH_PE_sdata4
// DWARF32:   DW_CFA_def_cfa: reg29 +0
// FIXME: The instructions are different from the ones produces by gas.
//
// DWARF32: 00000014 00000010 00000018 FDE cie=00000018 pc=00000000...00000000
// DWARF32:   DW_CFA_nop:
// DWARF32:   DW_CFA_nop:
// DWARF32:   DW_CFA_nop:

// DWARF64: 00000000 00000010 ffffffff CIE
// DWARF64:   Version:               1
// DWARF64:   Augmentation:          "zR"
// DWARF64:   Code alignment factor: 1
// DWARF64:   Data alignment factor: -8
//                                   ^^ GAS uses -4. Should be ok as long as
//                                      all offsets we need are a multiple of 8.
// DWARF64:   Return address column: 31
// DWARF64:   Augmentation data:     0C
//                                   ^^ fde pointer encoding: DW_EH_PE_sdata8
// DWARF64:   DW_CFA_def_cfa: reg29 +0
// FIXME: The instructions are different from the ones produces by gas.
//
// DWARF64: 00000014 00000018 00000018 FDE cie=00000018 pc=00000000...00000000
// DWARF64:   DW_CFA_nop:
// DWARF64:   DW_CFA_nop:
// DWARF64:   DW_CFA_nop:
